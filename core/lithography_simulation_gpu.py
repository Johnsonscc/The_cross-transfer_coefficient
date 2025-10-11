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

    # 检查Metal支持（Mac GPU）
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


def compute_tcc_4d_gpu(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):

    Lx, Ly = len(fx), len(fy)

    # 将数据转移到GPU（一次性）
    fx_tensor = torch.tensor(fx, dtype=torch.float32, device=device)
    fy_tensor = torch.tensor(fy, dtype=torch.float32, device=device)

    # 创建频率网格
    Fx, Fy = torch.meshgrid(fx_tensor, fy_tensor, indexing='ij')

    # 预计算光源函数
    J_grid = light_source_function_torch(Fx, Fy, sigma, na, lambda_)

    # 预计算所有可能的pupil函数值（在GPU上）
    print("在GPU上预计算pupil函数查找表...")
    pupil_cache_gpu = {}
    all_fx_vals = torch.unique(torch.cat([fx_tensor + fx_tensor[i] for i in range(Lx)]))
    all_fy_vals = torch.unique(torch.cat([fy_tensor + fy_tensor[i] for i in range(Ly)]))

    for fx_val in all_fx_vals:
        for fy_val in all_fy_vals:
            pupil_cache_gpu[(fx_val.item(), fy_val.item())] = pupil_response_function_torch(
                torch.tensor([fx_val]), torch.tensor([fy_val]), na, lambda_
            ).squeeze()

    # 初始化TCC矩阵
    tcc_4d = torch.zeros((Lx, Ly, Lx, Ly), dtype=torch.complex64, device=device)

    # 频率间隔
    df = fx[1] - fx[0]

    # 向量化计算
    print("使用向量化GPU计算...")
    total_iterations = Lx * Ly
    pbar = tqdm(total=total_iterations, desc="GPU TCC计算进度", unit="pixel")

    for i_prime in range(Lx):
        for j_prime in range(Ly):
            # 向量化计算P1_grid
            P1_grid = torch.zeros((Lx, Ly), dtype=torch.complex64, device=device)
            for m in range(Lx):
                for n in range(Ly):
                    key = ((fx[m] + fx[i_prime]), (fy[n] + fy[j_prime]))
                    P1_grid[m, n] = pupil_cache_gpu.get(key, pupil_response_function_torch(
                        torch.tensor([key[0]]), torch.tensor([key[1]]), na, lambda_
                    ).squeeze())

            for i_dprime in range(Lx):
                for j_dprime in range(Ly):
                    # 向量化计算P2_conj_grid
                    P2_conj_grid = torch.zeros((Lx, Ly), dtype=torch.complex64, device=device)
                    for m in range(Lx):
                        for n in range(Ly):
                            key = ((fx[m] + fx[i_dprime]), (fy[n] + fy[j_dprime]))
                            P2_conj_grid[m, n] = torch.conj(pupil_cache_gpu.get(key, pupil_response_function_torch(
                                torch.tensor([key[0]]), torch.tensor([key[1]]), na, lambda_
                            ).squeeze()))

                    # 向量化计算积分
                    integrand = J_grid * P1_grid * P2_conj_grid
                    tcc_4d[i_prime, j_prime, i_dprime, j_dprime] = torch.sum(integrand) * (df * df)

            pbar.update(1)

    pbar.close()
    return tcc_4d.cpu().numpy()

# 原有函数保持不变
def compute_tcc_svd(tcc_4d, num_modes=None):
    """原有SVD函数保持不变"""
    Lx, Ly, _, _ = tcc_4d.shape
    tcc_2d = tcc_4d.reshape(Lx * Ly, Lx * Ly)
    U, S, Vh = svd(tcc_2d, full_matrices=False)

    if num_modes is None:
        num_modes = np.sum(S > 1e-10)

    eigenvalues = S[:num_modes]
    eigenvectors = U[:, :num_modes].reshape(Lx, Ly, num_modes)
    return eigenvalues, eigenvectors


def socs_imaging(mask, eigenvalues, eigenvectors):
    """原有SOCS成像函数保持不变"""
    M_fft = fftshift(fft2(mask))
    I_xy = np.zeros_like(mask, dtype=float)

    for i in range(len(eigenvalues)):
        effective_mask_fft = M_fft * eigenvectors[:, :, i]
        amplitude = ifft2(ifftshift(effective_mask_fft))
        I_xy += eigenvalues[i] * np.abs(amplitude) ** 2

    return I_xy


# 修改主函数，使用GPU加速的TCC计算
def enhanced_hopkins_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                z=Z, dx=DX, dy=DY, n=N, sigma=SIGMA, na=NA,
                                method='socs', num_modes=10):
    """
    使用GPU加速的增强版Hopkins光刻仿真
    """
    # 计算频率坐标
    fx = np.fft.fftfreq(lx, dx)
    fy = np.fft.fftfreq(ly, dy)

    if method == 'socs':
        # 使用GPU加速计算TCC矩阵
        print("使用GPU加速计算TCC矩阵...")
        tcc_4d = compute_tcc_4d_gpu(fx, fy, sigma, na, lambda_)

        # 对TCC进行SVD分解
        print("进行TCC SVD分解...")
        eigenvalues, eigenvectors = compute_tcc_svd(tcc_4d, num_modes)

        # 使用SOCS算法计算光强
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