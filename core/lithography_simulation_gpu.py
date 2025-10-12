import numpy as np
import torch
import torch.fft as torch_fft
from tqdm import tqdm
from config.parameters import *

# 自动选择最佳设备
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU acceleration")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal) acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU (no GPU acceleration available)")


def light_source_function(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    """光源函数 - GPU版本"""
    r = torch.sqrt(fx ** 2 + fy ** 2)
    r_max = sigma * na / lambda_
    J = torch.where(r <= r_max, lambda_ ** 2 / (np.pi * (sigma * na) ** 2), 0.0)
    return J


def pupil_response_function(fx, fy, na=NA, lambda_=LAMBDA):
    """光瞳频率响应函数 - GPU版本"""
    r = torch.sqrt(fx ** 2 + fy ** 2)
    r_max = na / lambda_
    P = torch.where(r <= r_max, lambda_ ** 2 / (np.pi * na ** 2), 0.0)
    return P


def compute_tcc_4d_gpu(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    """
    计算四维TCC矩阵 - GPU加速版本
    添加进度条显示计算进度
    """
    Lx, Ly = len(fx), len(fy)

    # 将频率坐标转换为tensor并移到设备，确保使用float32
    fx_tensor = torch.from_numpy(fx.astype(np.float32)).to(device)
    fy_tensor = torch.from_numpy(fy.astype(np.float32)).to(device)

    # 预计算光源函数网格
    Fx, Fy = torch.meshgrid(fx_tensor, fy_tensor, indexing='ij')
    J_grid = light_source_function(Fx, Fy, sigma, na, lambda_)

    # 初始化四维TCC矩阵，使用complex64而不是complex128
    tcc = torch.zeros((Lx, Ly, Lx, Ly), dtype=torch.complex64, device=device)

    # 计算TCC - 使用进度条
    total_iterations = Lx * Ly * Lx * Ly
    progress_bar = tqdm(total=total_iterations, desc="Computing TCC matrix")

    for i_prime in range(Lx):
        for j_prime in range(Ly):
            for i_dprime in range(Lx):
                for j_dprime in range(Ly):
                    # 计算被积函数
                    integrand = torch.zeros((Lx, Ly), dtype=torch.complex64, device=device)

                    for m in range(Lx):
                        for n in range(Ly):
                            # P(f+f',g+g')
                            P_original = pupil_response_function(
                                fx_tensor[m] + fx_tensor[i_prime],
                                fy_tensor[n] + fy_tensor[j_prime],
                                na, lambda_
                            )

                            # P*(f+f'',g+g'')
                            P_conjugate = torch.conj(pupil_response_function(
                                fx_tensor[m] + fx_tensor[i_dprime],
                                fy_tensor[n] + fy_tensor[j_dprime],
                                na, lambda_
                            ))

                            integrand[m, n] = J_grid[m, n] * P_original * P_conjugate

                    # 数值积分 - 使用梯形法则
                    integral_y = torch.zeros(Lx, dtype=torch.complex64, device=device)
                    for i in range(Lx):
                        # 手动实现梯形积分
                        y_vals = integrand[i, :]
                        integral_y[i] = torch.trapz(y_vals, fy_tensor)

                    tcc_value = torch.trapz(integral_y, fx_tensor)
                    tcc[i_prime, j_prime, i_dprime, j_dprime] = tcc_value

                    progress_bar.update(1)

    progress_bar.close()
    return tcc.cpu().numpy()  # 移回CPU并转换为numpy


def compute_tcc_svd_gpu(tcc_4d, num_modes=10):
    """
    对4D TCC进行SVD分解 - GPU加速版本
    """
    Lx, Ly, _, _ = tcc_4d.shape

    # 将TCC移到GPU，确保使用正确的数据类型
    tcc_tensor = torch.from_numpy(tcc_4d.astype(np.complex64)).to(device)

    # 重塑为2D矩阵进行SVD
    tcc_2d = tcc_tensor.reshape(Lx * Ly, Lx * Ly)

    # SVD分解 - 对于复数矩阵，使用适当的SVD
    # 注意：对于复数矩阵，torch.linalg.svd返回U, S, Vh，其中Vh已经是共轭转置
    U, S, Vh = torch.linalg.svd(tcc_2d, full_matrices=False)

    # 取前num_modes个主要模式
    tcc_svs = S[:num_modes].cpu().numpy()
    eigen_vectors = U[:, :num_modes].reshape(Lx, Ly, num_modes).cpu().numpy()

    return tcc_svs, eigen_vectors


def socs_imaging_gpu(mask, tcc_svs, eigen_vectors):
    """
    SOCS算法实现 - GPU加速版本
    """
    # 将数据转移到GPU，确保使用正确的数据类型
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)
    tcc_svs_tensor = torch.from_numpy(tcc_svs.astype(np.float32)).to(device)
    eigen_vectors_tensor = torch.from_numpy(eigen_vectors.astype(np.complex64)).to(device)

    # 计算掩模频谱
    M_fft = torch_fft.fftshift(torch_fft.fft2(mask_tensor))

    # 初始化光强
    I_xy = torch.zeros_like(mask_tensor, dtype=torch.float32, device=device)

    # 对每个特征系统求和
    for i, (sigma, H_i) in enumerate(zip(tcc_svs_tensor, eigen_vectors_tensor.permute(2, 0, 1))):
        # 每个相干系统的有效掩模
        effective_mask_fft = M_fft * H_i

        # 相干成像
        coherent_image = torch.abs(torch_fft.ifft2(torch_fft.ifftshift(effective_mask_fft))) ** 2

        # 加权求和
        I_xy += sigma * coherent_image

    # 移回CPU并转换为numpy
    return I_xy.cpu().numpy()


def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                           z=Z, dx=DX, dy=DY, n=N, sigma=SIGMA, na=NA,
                                           num_modes=10, precomputed_tcc=None):
    """
    使用SOCS算法的Hopkins光刻仿真 - GPU加速版本

    参数:
    - mask: 输入掩膜
    - precomputed_tcc: 预计算的TCC矩阵，如果为None则重新计算
    """
    # 计算频率坐标
    fx = np.fft.fftfreq(lx, dx).astype(np.float32)  # 确保使用float32
    fy = np.fft.fftfreq(ly, dy).astype(np.float32)

    # 计算或使用预计算的TCC
    if precomputed_tcc is None:
        print("Computing TCC matrix...")
        tcc_4d = compute_tcc_4d_gpu(fx, fy, sigma, na, lambda_)
    else:
        tcc_4d = precomputed_tcc

    # SVD分解
    print("Performing SVD decomposition...")
    tcc_svs, eigen_vectors = compute_tcc_svd_gpu(tcc_4d, num_modes)

    # 使用SOCS算法进行成像
    print("Performing SOCS imaging...")
    result = socs_imaging_gpu(mask, tcc_svs, eigen_vectors)

    return result, tcc_4d  # 返回结果和TCC以便后续使用


# 预计算TCC的函数（用于多次仿真）
def precompute_tcc(lambda_=LAMBDA, lx=LX, ly=LY, dx=DX, dy=DY, sigma=SIGMA, na=NA):
    """预计算TCC矩阵以便多次使用"""
    fx = np.fft.fftfreq(lx, dx).astype(np.float32)
    fy = np.fft.fftfreq(ly, dy).astype(np.float32)
    return compute_tcc_4d_gpu(fx, fy, sigma, na, lambda_)