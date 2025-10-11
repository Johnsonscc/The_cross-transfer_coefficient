import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.linalg import svd
from config.parameters import *
from tqdm import tqdm

# 添加GPU支持（仅用于TCC计算）
try:
    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Metal GPU加速可用，TCC计算将使用GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ CUDA GPU加速可用，TCC计算将使用GPU")
    else:
        device = torch.device("cpu")
        print("❌ 未检测到GPU，TCC计算将使用CPU")

    GPU_AVAILABLE = True
except ImportError:
    print("❌ PyTorch未安装，TCC计算将使用CPU")
    GPU_AVAILABLE = False
    device = None


def transfer_function(fx, fy, lambda_=LAMBDA, z=Z, n=N):
    H = np.exp(-1j * np.pi * lambda_ * z * (fx ** 2 + fy ** 2) / n ** 2)
    return H


# 曝光光源强度函数
def light_source_function(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = sigma * na / lambda_
    J = np.where(r <= r_max, lambda_ ** 2 / (np.pi * (sigma * na) ** 2), 0)
    return J


# 光瞳频率响应函数
def pupil_response_function(fx, fy, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = na / lambda_
    P = np.where(r <= r_max, lambda_ ** 2 / (np.pi * (na) ** 2), 0)
    return P


# GPU加速的光源函数
def light_source_function_torch(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    r = torch.sqrt(fx ** 2 + fy ** 2)
    r_max = sigma * na / lambda_
    J = torch.where(r <= r_max, torch.tensor(lambda_ ** 2 / (np.pi * (sigma * na) ** 2)), torch.tensor(0.0))
    return J


# GPU加速的光瞳函数
def pupil_response_function_torch(fx, fy, na=NA, lambda_=LAMBDA):
    r = torch.sqrt(fx ** 2 + fy ** 2)
    r_max = na / lambda_
    P = torch.where(r <= r_max, torch.tensor(lambda_ ** 2 / (np.pi * (na) ** 2)), torch.tensor(0.0))
    return P


def compute_tcc_4d_gpu(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA, block_size=8):
    """
    GPU加速的TCC计算
    """
    Lx, Ly = len(fx), len(fy)

    # 将数据转移到GPU
    fx_tensor = torch.tensor(fx, dtype=torch.float32, device=device)
    fy_tensor = torch.tensor(fy, dtype=torch.float32, device=device)

    # 创建频率网格
    Fx, Fy = torch.meshgrid(fx_tensor, fy_tensor, indexing='ij')

    # 预计算光源函数
    J_grid = light_source_function_torch(Fx, Fy, sigma, na, lambda_)

    # 初始化TCC矩阵
    tcc_4d = torch.zeros((Lx, Ly, Lx, Ly), dtype=torch.complex64, device=device)

    # 频率间隔
    df = fx[1] - fx[0]

    # 分块计算以避免内存溢出
    total_blocks = (Lx // block_size) * (Ly // block_size)
    pbar = tqdm(total=total_blocks, desc="GPU TCC计算进度", unit="block")

    for i_prime in range(0, Lx, block_size):
        for j_prime in range(0, Ly, block_size):
            i_end = min(i_prime + block_size, Lx)
            j_end = min(j_prime + block_size, Ly)

            # 预计算P1_grid对于当前块
            P1_grid = torch.zeros((i_end - i_prime, j_end - j_prime, Lx, Ly),
                                  dtype=torch.complex64, device=device)

            for i_local, i_global in enumerate(range(i_prime, i_end)):
                for j_local, j_global in enumerate(range(j_prime, j_end)):
                    for m in range(Lx):
                        for n in range(Ly):
                            P1_grid[i_local, j_local, m, n] = pupil_response_function_torch(
                                fx_tensor[m] + fx_tensor[i_global],
                                fy_tensor[n] + fy_tensor[j_global],
                                na, lambda_
                            )

            # 计算当前块的TCC
            for i_dprime in range(Lx):
                for j_dprime in range(Ly):
                    # 计算P2_conj_grid
                    P2_conj_grid = torch.zeros((Lx, Ly), dtype=torch.complex64, device=device)
                    for m in range(Lx):
                        for n in range(Ly):
                            P2_conj_grid[m, n] = torch.conj(pupil_response_function_torch(
                                fx_tensor[m] + fx_tensor[i_dprime],
                                fy_tensor[n] + fy_tensor[j_dprime],
                                na, lambda_
                            ))

                    # 计算当前块的被积函数
                    for i_local, i_global in enumerate(range(i_prime, i_end)):
                        for j_local, j_global in enumerate(range(j_prime, j_end)):
                            integrand = J_grid * P1_grid[i_local, j_local] * P2_conj_grid
                            tcc_4d[i_global, j_global, i_dprime, j_dprime] = torch.sum(integrand) * (df * df)

            pbar.update(1)

    pbar.close()
    return tcc_4d.cpu().numpy()


def compute_tcc_4d(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA, use_gpu=True):
    """
    优化的TCC计算，使用向量化操作减少循环，并显示进度条
    """
    # 如果GPU可用且问题规模较大，使用GPU版本
    if use_gpu and GPU_AVAILABLE and len(fx) * len(fy) > 500:  # 阈值可根据需要调整
        print("使用GPU加速计算TCC矩阵")
        return compute_tcc_4d_gpu(fx, fy, sigma, na, lambda_)

    Lx, Ly = len(fx), len(fy)

    # 创建频率网格
    Fx, Fy = np.meshgrid(fx, fy, indexing='ij')

    # 预计算光源函数
    J_grid = light_source_function(Fx, Fy, sigma, na, lambda_)

    # 预计算光瞳函数在所有可能偏移处的值
    P_cache = {}

    def get_pupil_value(fx_val, fy_val):
        key = (fx_val, fy_val)
        if key not in P_cache:
            P_cache[key] = pupil_response_function(fx_val, fy_val, na, lambda_)
        return P_cache[key]

    # 初始化TCC矩阵
    tcc_4d = np.zeros((Lx, Ly, Lx, Ly), dtype=complex)

    # 使用更高效的循环结构
    df = fx[1] - fx[0]  # 频率间隔

    # 创建总进度条
    total_iterations = Lx * Ly
    pbar_outer = tqdm(total=total_iterations, desc="CPU TCC计算进度", unit="bit")

    # 外层循环优化
    for i_prime in range(Lx):
        for j_prime in range(Ly):
            # 预计算P(f+f',g+g')对于所有m,n
            P1_grid = np.zeros((Lx, Ly), dtype=complex)
            for m in range(Lx):
                for n in range(Ly):
                    P1_grid[m, n] = get_pupil_value(
                        fx[m] + fx[i_prime],
                        fy[n] + fy[j_prime]
                    )

            for i_dprime in range(Lx):
                for j_dprime in range(Ly):
                    # 预计算P*(f+f'',g+g'')对于所有m,n
                    P2_conj_grid = np.zeros((Lx, Ly), dtype=complex)
                    for m in range(Lx):
                        for n in range(Ly):
                            P2_conj_grid[m, n] = np.conjugate(get_pupil_value(
                                fx[m] + fx[i_dprime],
                                fy[n] + fy[j_dprime]
                            ))

                    # 计算被积函数并积分
                    integrand = J_grid * P1_grid * P2_conj_grid
                    tcc_4d[i_prime, j_prime, i_dprime, j_dprime] = np.sum(integrand) * (df * df)

            # 更新进度条
            pbar_outer.update(1)

    pbar_outer.close()
    return tcc_4d


# 以下所有其他函数保持不变...

def compute_tcc_svd(tcc_4d, num_modes=None):
    """
    对4D TCC进行SVD分解，得到SOCS模型的特征值和特征函数
    """
    Lx, Ly, _, _ = tcc_4d.shape

    # 将4D TCC重塑为2D矩阵
    tcc_2d = tcc_4d.reshape(Lx * Ly, Lx * Ly)

    # 进行SVD分解
    U, S, Vh = svd(tcc_2d, full_matrices=False)

    # 如果未指定模式数量，使用所有非零特征值
    if num_modes is None:
        num_modes = np.sum(S > 1e-10)

    # 取前num_modes个主要模式
    eigenvalues = S[:num_modes]
    eigenvectors = U[:, :num_modes].reshape(Lx, Ly, num_modes)

    return eigenvalues, eigenvectors


def socs_imaging(mask, eigenvalues, eigenvectors):
    """
    使用SOCS算法进行成像计算
    I(x,y) = Σ σ_i |F^{-1}[M(f,g) * φ_i(f,g)]|^2
    """
    # 计算掩模频谱
    M_fft = fftshift(fft2(mask))

    # 初始化光强
    I_xy = np.zeros_like(mask, dtype=float)

    # 对每个特征模式求和
    for i in range(len(eigenvalues)):
        # 有效掩模频谱 = 原始掩模频谱 × 特征函数
        effective_mask_fft = M_fft * eigenvectors[:, :, i]

        # 逆傅里叶变换得到空间域振幅
        amplitude = ifft2(ifftshift(effective_mask_fft))

        # 计算光强并加权求和
        I_xy += eigenvalues[i] * np.abs(amplitude) ** 2

    return I_xy


def hopkins_imaging_spatial_domain(mask, tcc_4d, x_coords, y_coords):
    """
    空间域Hopkins成像模型
    I(x,y) = ∬∬ TCC(f',g',f'',g'') M(f',g') M*(f'',g'') e^{-j2π(x(f'-f'')+y(g'-g''))} df'dg'df''dg''
    """
    Lx, Ly = mask.shape

    # 计算掩模频谱
    M_fft = fftshift(fft2(mask))

    # 频率坐标
    fx = np.fft.fftfreq(Lx, DX)
    fy = np.fft.fftfreq(Ly, DY)

    # 初始化空间域光强
    I_xy = np.zeros((len(y_coords), len(x_coords)))

    # 创建空间坐标网格
    X, Y = np.meshgrid(x_coords, y_coords)

    # 计算空间域光强（简化版，实际应用需要优化性能）
    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            x, y = x_coords[i], y_coords[j]
            intensity = 0

            # 四重积分（简化计算，实际应用需要优化）
            for i_prime in range(Lx):
                for j_prime in range(Ly):
                    for i_dprime in range(Lx):
                        for j_dprime in range(Ly):
                            # 相位因子
                            phase_factor = np.exp(-2j * np.pi * (
                                    x * (fx[i_prime] - fx[i_dprime]) +
                                    y * (fy[j_prime] - fy[j_dprime])
                            ))

                            intensity += (
                                    tcc_4d[i_prime, j_prime, i_dprime, j_dprime] *
                                    M_fft[i_prime, j_prime] *
                                    np.conj(M_fft[i_dprime, j_dprime]) *
                                    phase_factor
                            )

            I_xy[j, i] = np.real(intensity)

    return I_xy


def enhanced_hopkins_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                z=Z, dx=DX, dy=DY, n=N, sigma=SIGMA, na=NA,
                                method='socs', num_modes=10, use_gpu=True):
    """
    增强版Hopkins光刻仿真，支持SOCS算法和空间域模型

    参数:
    mask: 输入掩模
    method: 成像方法，'socs'或'spatial'
    num_modes: SOCS方法中使用的特征模式数量
    use_gpu: 是否使用GPU加速TCC计算
    """

    # 计算频率坐标
    fx = np.fft.fftfreq(lx, dx)
    fy = np.fft.fftfreq(ly, dy)

    if method == 'socs':
        # 使用SOCS算法

        # 计算TCC矩阵
        print("计算TCC矩阵...")
        tcc_4d = compute_tcc_4d(fx, fy, sigma, na, lambda_, use_gpu)

        # 对TCC进行SVD分解
        print("进行TCC SVD分解...")
        eigenvalues, eigenvectors = compute_tcc_svd(tcc_4d, num_modes)

        # 使用SOCS算法计算光强
        print("使用SOCS算法计算光强...")
        I_xy = socs_imaging(mask, eigenvalues, eigenvectors)

    elif method == 'spatial':
        # 使用空间域Hopkins模型

        # 计算TCC矩阵
        print("计算TCC矩阵...")
        tcc_4d = compute_tcc_4d(fx, fy, sigma, na, lambda_, use_gpu)

        # 空间坐标
        x_coords = np.arange(-lx // 2, lx // 2) * dx
        y_coords = np.arange(-ly // 2, ly // 2) * dy

        # 使用空间域模型计算光强
        print("使用空间域Hopkins模型计算光强...")
        I_xy = hopkins_imaging_spatial_domain(mask, tcc_4d, x_coords, y_coords)

    else:
        raise ValueError("不支持的成像方法。请选择'socs'或'spatial'")

    return I_xy


# 保持向后兼容性
def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                           z=Z, dx=DX, dy=DY, n=N, sigma=SIGMA, na=NA,
                                           method='socs', num_modes=10, use_gpu=True):
    """
    兼容原有接口的增强版光刻仿真
    """
    return enhanced_hopkins_simulation(mask, lambda_, lx, ly, z, dx, dy, n, sigma, na, method, num_modes, use_gpu)