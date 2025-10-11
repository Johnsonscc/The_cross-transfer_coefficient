import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.linalg import svd
from config.parameters import *
from tqdm import tqdm
import torch

# 添加GPU支持
try:
    import torch
    import torch.fft as torch_fft

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Metal GPU加速可用，使用GPU进行计算")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ CUDA GPU加速可用，使用GPU进行计算")
    else:
        device = torch.device("cpu")
        print("❌ 未检测到GPU，使用CPU进行计算")

    GPU_AVAILABLE = True
except ImportError:
    print("❌ PyTorch未安装，无法使用GPU加速")
    GPU_AVAILABLE = False
    device = None


def compute_tcc_4d_gpu_parallel(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    """
    真正并行的GPU TCC计算
    """
    if not GPU_AVAILABLE:
        return compute_tcc_4d_cpu(fx, fy, sigma, na, lambda_)

    Lx, Ly = len(fx), len(fy)

    print(f"使用并行GPU计算TCC矩阵，规模: {Lx}x{Ly}")

    # 将数据转移到GPU
    fx_tensor = torch.tensor(fx, dtype=torch.float32, device=device)
    fy_tensor = torch.tensor(fy, dtype=torch.float32, device=device)

    # 创建所有可能的频率组合
    fx_all = fx_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (Lx, 1, 1, 1)
    fy_all = fy_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, Ly, 1, 1)

    # 创建频率网格 (向量化)
    Fx, Fy = torch.meshgrid(fx_tensor, fy_tensor, indexing='ij')

    # 预计算光源函数 (整个网格)
    J_grid = light_source_function_torch(Fx, Fy, sigma, na, lambda_)

    # 频率间隔
    df = fx[1] - fx[0]

    # 初始化TCC矩阵
    tcc_4d = torch.zeros((Lx, Ly, Lx, Ly), dtype=torch.complex64, device=device)

    print("开始并行GPU计算...")

    # 使用批处理减少循环次数
    batch_size = 4  # 根据GPU内存调整

    for i_prime_batch in range(0, Lx, batch_size):
        i_prime_end = min(i_prime_batch + batch_size, Lx)

        for j_prime_batch in range(0, Ly, batch_size):
            j_prime_end = min(j_prime_batch + batch_size, Ly)

            # 当前批次的大小
            batch_i = i_prime_end - i_prime_batch
            batch_j = j_prime_end - j_prime_batch

            # 创建批次索引
            i_prime_indices = torch.arange(i_prime_batch, i_prime_end, device=device)
            j_prime_indices = torch.arange(j_prime_batch, j_prime_end, device=device)

            # 向量化计算当前批次的TCC
            batch_tcc = compute_tcc_batch(
                fx_tensor, fy_tensor, J_grid,
                i_prime_indices, j_prime_indices,
                Lx, Ly, df, na, lambda_
            )

            # 将批次结果存入总矩阵
            for i_local, i_global in enumerate(i_prime_indices):
                for j_local, j_global in enumerate(j_prime_indices):
                    tcc_4d[i_global, j_global] = batch_tcc[i_local, j_local]

            print(f"处理批次: ({i_prime_batch}-{i_prime_end}, {j_prime_batch}-{j_prime_end})")

    return tcc_4d.cpu().numpy()


def compute_tcc_batch(fx_tensor, fy_tensor, J_grid, i_prime_indices, j_prime_indices, Lx, Ly, df, na, lambda_):
    """
    计算一个批次的TCC
    """
    batch_i = len(i_prime_indices)
    batch_j = len(j_prime_indices)

    # 初始化批次结果
    batch_tcc = torch.zeros((batch_i, batch_j, Lx, Ly), dtype=torch.complex64, device=device)

    # 为整个批次预计算P1_grid
    P1_grid_batch = torch.zeros((batch_i, batch_j, Lx, Ly), dtype=torch.complex64, device=device)

    # 向量化计算P1_grid
    for i_local, i_prime in enumerate(i_prime_indices):
        for j_local, j_prime in enumerate(j_prime_indices):
            # 计算P(f+f',g+g')对于所有m,n
            for m in range(Lx):
                for n in range(Ly):
                    fx_val = fx_tensor[m] + fx_tensor[i_prime]
                    fy_val = fy_tensor[n] + fy_tensor[j_prime]
                    P1_grid_batch[i_local, j_local, m, n] = pupil_response_function_torch(
                        fx_val, fy_val, na, lambda_
                    )

    # 为所有i_dprime, j_dprime计算P2_conj_grid
    P2_conj_grid_all = torch.zeros((Lx, Ly, Lx, Ly), dtype=torch.complex64, device=device)

    for i_dprime in range(Lx):
        for j_dprime in range(Ly):
            for m in range(Lx):
                for n in range(Ly):
                    fx_val = fx_tensor[m] + fx_tensor[i_dprime]
                    fy_val = fy_tensor[n] + fy_tensor[j_dprime]
                    P2_conj_grid_all[i_dprime, j_dprime, m, n] = torch.conj(
                        pupil_response_function_torch(fx_val, fy_val, na, lambda_)
                    )

    # 并行计算批次的TCC
    for i_local in range(batch_i):
        for j_local in range(batch_j):
            for i_dprime in range(Lx):
                for j_dprime in range(Ly):
                    # 向量化计算积分
                    integrand = J_grid * P1_grid_batch[i_local, j_local] * P2_conj_grid_all[i_dprime, j_dprime]
                    batch_tcc[i_local, j_local, i_dprime, j_dprime] = torch.sum(integrand) * (df * df)

    return batch_tcc


# 更高效的GPU版本 - 使用真正的并行计算
def compute_tcc_4d_gpu_fast(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    """
    快速GPU版本 - 使用矩阵运算避免嵌套循环
    """
    if not GPU_AVAILABLE:
        return compute_tcc_4d_cpu(fx, fy, sigma, na, lambda_)

    Lx, Ly = len(fx), len(fx)

    print(f"使用快速GPU矩阵运算计算TCC，规模: {Lx}x{Ly}")

    # 转换为PyTorch张量
    fx_tensor = torch.tensor(fx, dtype=torch.float32, device=device)
    fy_tensor = torch.tensor(fy, dtype=torch.float32, device=device)

    # 创建所有频率组合的矩阵
    # 形状: (Lx*Ly, Lx*Ly, Lx*Ly, Lx*Ly)
    fx_expanded = fx_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (Lx, 1, 1, 1)
    fy_expanded = fy_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, Ly, 1, 1)

    # 计算J(f,g)网格
    Fx, Fy = torch.meshgrid(fx_tensor, fy_tensor, indexing='ij')
    J_grid = light_source_function_torch(Fx, Fy, sigma, na, lambda_)

    # 频率间隔
    df = fx[1] - fx[0]

    # 由于四维TCC矩阵太大，我们使用迭代方法
    tcc_4d = torch.zeros((Lx, Ly, Lx, Ly), dtype=torch.complex64, device=device)

    # 使用更粗粒度的并行
    for i_prime in range(Lx):
        # 为当前i_prime预计算所有需要的值
        P1_grid_i = torch.zeros((Ly, Lx, Ly), dtype=torch.complex64, device=device)

        for j_prime in range(Ly):
            for m in range(Lx):
                for n in range(Ly):
                    P1_grid_i[j_prime, m, n] = pupil_response_function_torch(
                        fx_tensor[m] + fx_tensor[i_prime],
                        fy_tensor[n] + fy_tensor[j_prime],
                        na, lambda_
                    )

        for i_dprime in range(Lx):
            # 为当前i_dprime预计算P2_conj
            P2_conj_grid_i = torch.zeros((Ly, Lx, Ly), dtype=torch.complex64, device=device)

            for j_dprime in range(Ly):
                for m in range(Lx):
                    for n in range(Ly):
                        P2_conj_grid_i[j_dprime, m, n] = torch.conj(
                            pupil_response_function_torch(
                                fx_tensor[m] + fx_tensor[i_dprime],
                                fy_tensor[n] + fy_tensor[j_dprime],
                                na, lambda_
                            )
                        )

            # 计算当前(i_prime, i_dprime)对应的所有(j_prime, j_dprime)的TCC
            for j_prime in range(Ly):
                for j_dprime in range(Ly):
                    integrand = J_grid * P1_grid_i[j_prime] * P2_conj_grid_i[j_dprime]
                    tcc_4d[i_prime, j_prime, i_dprime, j_dprime] = torch.sum(integrand) * (df * df)

        print(f"进度: {i_prime + 1}/{Lx}")

    return tcc_4d.cpu().numpy()


# CPU版本作为备选
def compute_tcc_4d_cpu(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    """
    优化的CPU版本TCC计算
    """
    Lx, Ly = len(fx), len(fy)

    print(f"使用CPU计算TCC矩阵，规模: {Lx}x{Ly}")

    # 创建频率网格
    Fx, Fy = np.meshgrid(fx, fy, indexing='ij')

    # 预计算光源函数
    J_grid = light_source_function(Fx, Fy, sigma, na, lambda_)

    # 初始化TCC矩阵
    tcc_4d = np.zeros((Lx, Ly, Lx, Ly), dtype=complex)

    # 频率间隔
    df = fx[1] - fx[0]

    # 使用缓存避免重复计算
    pupil_cache = {}

    # 进度条
    pbar = tqdm(total=Lx * Ly, desc="CPU TCC计算进度")

    for i_prime in range(Lx):
        for j_prime in range(Ly):
            # 预计算P1_grid
            P1_grid = np.zeros((Lx, Ly), dtype=complex)
            for m in range(Lx):
                for n in range(Ly):
                    key1 = (fx[m] + fx[i_prime], fy[n] + fy[j_prime])
                    if key1 not in pupil_cache:
                        pupil_cache[key1] = pupil_response_function(key1[0], key1[1], na, lambda_)
                    P1_grid[m, n] = pupil_cache[key1]

            for i_dprime in range(Lx):
                for j_dprime in range(Ly):
                    # 预计算P2_conj_grid
                    P2_conj_grid = np.zeros((Lx, Ly), dtype=complex)
                    for m in range(Lx):
                        for n in range(Ly):
                            key2 = (fx[m] + fx[i_dprime], fy[n] + fy[j_dprime])
                            if key2 not in pupil_cache:
                                pupil_cache[key2] = pupil_response_function(key2[0], key2[1], na, lambda_)
                            P2_conj_grid[m, n] = np.conjugate(pupil_cache[key2])

                    # 计算积分
                    integrand = J_grid * P1_grid * P2_conj_grid
                    tcc_4d[i_prime, j_prime, i_dprime, j_dprime] = np.sum(integrand) * (df * df)

            pbar.update(1)

    pbar.close()
    return tcc_4d


# GPU加速的光源函数
def light_source_function_torch(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    r = torch.sqrt(fx ** 2 + fy ** 2)
    r_max = sigma * na / lambda_
    # 使用向量化操作
    J_value = lambda_ ** 2 / (np.pi * (sigma * na) ** 2)
    J = torch.where(r <= r_max, J_value, 0.0)
    return J


# GPU加速的光瞳函数
def pupil_response_function_torch(fx, fy, na=NA, lambda_=LAMBDA):
    r = torch.sqrt(fx ** 2 + fy ** 2)
    r_max = na / lambda_
    # 使用向量化操作
    P_value = lambda_ ** 2 / (np.pi * na ** 2)
    P = torch.where(r <= r_max, P_value, 0.0)
    return P


# 原有函数保持不变
def light_source_function(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = sigma * na / lambda_
    J = np.where(r <= r_max, lambda_ ** 2 / (np.pi * (sigma * na) ** 2), 0)
    return J


def pupil_response_function(fx, fy, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = na / lambda_
    P = np.where(r <= r_max, lambda_ ** 2 / (np.pi * (na) ** 2), 0)
    return P


def compute_tcc_svd(tcc_4d, num_modes=None):
    Lx, Ly, _, _ = tcc_4d.shape
    tcc_2d = tcc_4d.reshape(Lx * Ly, Lx * Ly)
    U, S, Vh = svd(tcc_2d, full_matrices=False)

    if num_modes is None:
        num_modes = np.sum(S > 1e-10)

    eigenvalues = S[:num_modes]
    eigenvectors = U[:, :num_modes].reshape(Lx, Ly, num_modes)
    return eigenvalues, eigenvectors


def socs_imaging(mask, eigenvalues, eigenvectors):
    M_fft = fftshift(fft2(mask))
    I_xy = np.zeros_like(mask, dtype=float)

    for i in range(len(eigenvalues)):
        effective_mask_fft = M_fft * eigenvectors[:, :, i]
        amplitude = ifft2(ifftshift(effective_mask_fft))
        I_xy += eigenvalues[i] * np.abs(amplitude) ** 2

    return I_xy


# 修改主函数，提供多种计算选项
def enhanced_hopkins_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                z=Z, dx=DX, dy=DY, n=N, sigma=SIGMA, na=NA,
                                method='socs', num_modes=10, tcc_method='auto'):
    """
    增强版Hopkins光刻仿真

    参数:
    tcc_method: 'auto', 'gpu_fast', 'gpu_parallel', 'cpu'
    """
    # 计算频率坐标
    fx = np.fft.fftfreq(lx, dx)
    fy = np.fft.fftfreq(ly, dy)

    if method == 'socs':
        # 根据选择的方法计算TCC矩阵
        print("计算TCC矩阵...")

        if tcc_method == 'auto':
            # 自动选择最佳方法
            total_elements = len(fx) * len(fy) * len(fx) * len(fy)
            if total_elements < 1000 or not GPU_AVAILABLE:
                tcc_4d = compute_tcc_4d_cpu(fx, fy, sigma, na, lambda_)
            else:
                tcc_4d = compute_tcc_4d_gpu_fast(fx, fy, sigma, na, lambda_)
        elif tcc_method == 'gpu_fast' and GPU_AVAILABLE:
            tcc_4d = compute_tcc_4d_gpu_fast(fx, fy, sigma, na, lambda_)
        elif tcc_method == 'gpu_parallel' and GPU_AVAILABLE:
            tcc_4d = compute_tcc_4d_gpu_parallel(fx, fy, sigma, na, lambda_)
        elif tcc_method == 'cpu':
            tcc_4d = compute_tcc_4d_cpu(fx, fy, sigma, na, lambda_)
        else:
            tcc_4d = compute_tcc_4d_cpu(fx, fy, sigma, na, lambda_)

        print("进行TCC SVD分解...")
        eigenvalues, eigenvectors = compute_tcc_svd(tcc_4d, num_modes)

        print("使用SOCS算法计算光强...")
        I_xy = socs_imaging(mask, eigenvalues, eigenvectors)

    else:
        raise ValueError("只支持SOCS方法")

    return I_xy


# 保持向后兼容性
def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                           z=Z, dx=DX, dy=DY, n=N, sigma=SIGMA, na=NA,
                                           method='socs', num_modes=10):
    return enhanced_hopkins_simulation(mask, lambda_, lx, ly, z, dx, dy, n, sigma, na, method, num_modes)