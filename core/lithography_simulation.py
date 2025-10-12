import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from config.parameters import *
from scipy.sparse.linalg import svds
from tqdm import tqdm  # 添加进度条库

def light_source_function(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = sigma * na / lambda_
    J = np.where(r <= r_max, lambda_ ** 2 / (np.pi * (sigma * na) ** 2), 0)
    return J


def pupil_response_function(fx, fy, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max=na / lambda_
    P = np.where(r < r_max, lambda_ ** 2 / (np.pi * (na) ** 2), 0)
    return P


def compute_tcc_svd(J, P, fx, fy, k):

    # 创建频域网格
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # 计算有效光源和瞳函数
    J_vals = J(FX, FY)
    P_vals = P(FX, FY)

    # 计算TCC核函数
    tcc_kernel = J_vals * P_vals
    Lx, Ly = len(fx), len(fy)
    TCC_4d = np.zeros((Lx, Ly, Lx, Ly), dtype=np.complex128)

    print("Building TCC matrix...")

    # 使用tqdm添加进度条
    for i in tqdm(range(Lx), desc="TCC Computation"):
        for j in range(Ly):
            for m in range(Lx):
                for n in range(Ly):
                    if (0 <= i - m < Lx) and (0 <= j - n < Ly):
                        TCC_4d[i, j, m, n] = tcc_kernel[i, j] * np.conj(tcc_kernel[m, n])

    print("Performing SVD decomposition...")

    # 重塑为2D矩阵进行SVD
    TCC_2d = TCC_4d.reshape(Lx * Ly, Lx * Ly)

    # 奇异值分解
    U, S, Vh = svds(TCC_2d, k=min(k, min(TCC_2d.shape) - 1))

    # 确保奇异值按降序排列
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]
    H_functions = []

    for i in tqdm(range(len(S)), desc="Extracting eigenfunctions"):
        H_i = U[:, i].reshape(Lx, Ly)
        H_functions.append(H_i)

    return S, H_functions


def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                            dx=DX, dy=DY, sigma=SIGMA, na=NA, k_svd=10):

    # 频域坐标
    fx = np.linspace(-0.5 / dx, 0.5 / dx, lx)
    fy = np.linspace(-0.5 / dy, 0.5 / dy, ly)

    # 定义光源和瞳函数
    J = lambda fx, fy: light_source_function(fx, fy, sigma, na, lambda_)
    P = lambda fx, fy: pupil_response_function(fx, fy, na, lambda_)

    # 计算TCC并进行SVD分解
    print("Computing TCC and performing SVD...")
    singular, H_functions = compute_tcc_svd(J, P, fx, fy, k_svd)

    # 掩模的傅里叶变换
    M_fft = fftshift(fft2(mask))

    # 初始化光强
    intensity = np.zeros((lx, ly), dtype=np.float64)

    # 根据SVD分解计算光强
    print(f"Computing intensity using {len(singular)} singular values...")
    for i, (s_val, H_i) in enumerate(zip(singular, H_functions)):
        # 滤波后的频谱
        filtered_fft = M_fft * H_i

        # 逆傅里叶变换
        filtered_space = ifft2(ifftshift(filtered_fft))

        # 累加光强贡献
        intensity += s_val * np.abs(filtered_space) ** 2


    # 最终结果
    result_fft = fft2(intensity)
    result = np.abs(ifft2(result_fft))

    # 归一化到[0,1]范围
    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result

