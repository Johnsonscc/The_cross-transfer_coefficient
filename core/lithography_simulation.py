# lithography_simulation_source.py
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.integrate import simpson
from scipy.signal import convolve2d
from config.parameters import *


def dmd_modulation(M_xy, Wx=WX, Wy=WY, Tx=TX, Ty=TY, dx=DX, dy=DY):
    """
    DMD调制掩模函数
    """
    M, N = M_xy.shape

    # 计算每个方向上的采样点数
    nx = int(np.round(Tx / dx))
    ny = int(np.round(Ty / dy))

    # 1. 构建梳状函数 comb(x/Tx, y/Ty)
    comb_x = np.zeros(nx)
    comb_y = np.zeros(ny)

    # 在周期位置设置脉冲
    comb_x[0] = 1  # 第一个位置
    comb_y[0] = 1  # 第一个位置

    # 构建二维梳状函数(外积)
    comb_xy = np.outer(comb_x, comb_y)

    # 2. 构建矩形函数 rect(x/Wx, y/Wy)
    # 计算矩形函数的尺寸
    wx_samples = int(np.round(Wx / dx))
    wy_samples = int(np.round(Wy / dy))

    # 创建矩形函数
    rect_x = np.zeros(nx)
    start_x = max(0, int((nx - wx_samples) / 2))
    end_x = min(nx, start_x + wx_samples)
    rect_x[start_x:end_x] = 1

    rect_y = np.zeros(ny)
    start_y = max(0, int((ny - wy_samples) / 2))
    end_y = min(ny, start_y + wy_samples)
    rect_y[start_y:end_y] = 1

    # 构建二维矩形函数
    rect_xy = np.outer(rect_x, rect_y)

    # 3. 构建调制函数 (1/Tx*Ty) * [rect * comb]
    # 使用二维卷积
    modulated_rc = 1 / (Tx * Ty) * convolve2d(rect_xy, comb_xy, mode='same')

    # 4. 掩模M(x,y)调制
    expand_rows = M * nx
    expand_cols = N * ny
    expand_M = np.zeros((expand_rows, expand_cols))

    for i in range(M):
        for j in range(N):
            start_row = i * nx
            start_col = j * ny
            end_row = start_row + nx
            end_col = start_col + ny
            # 调制窗口
            expand_M[start_row:end_row, start_col:end_col] = M_xy[i, j] * modulated_rc

    return expand_M


def compute_frequency(dx, dy, Lx, Ly):
    """
    计算频域坐标
    """
    # 创建频率坐标轴
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), Lx)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), Ly)
    # 创建二维频率网格
    return np.meshgrid(fx, fy, indexing='ij')


def light_source_function(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    """
    光源函数
    """
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = sigma * na / lambda_
    J = np.where(r <= r_max, lambda_ ** 2 / (np.pi * (sigma * na) ** 2), 0)
    return J


def pupil_response_function(fx, fy, na=NA, lambda_=LAMBDA):
    """
    光瞳响应函数
    """
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = na / lambda_
    P = np.where(r <= r_max, lambda_ ** 2 / (np.pi * na ** 2), 0)
    return P


def compute_tcc_4d(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    """
    计算四维TCC（交叉传递系数）
    """
    # 初始化四维矩阵
    Lx, Ly = len(fx), len(fy)
    tcc = np.zeros((Lx, Ly, Lx, Ly), dtype=complex)

    # 网格化光源函数和光瞳函数
    J_grid = np.zeros((Lx, Ly))
    P_grid = np.zeros((Lx, Ly), dtype=complex)

    # 创建二维频率网格
    Fx, Fy = np.meshgrid(fx, fy, indexing='ij')

    J_grid = light_source_function(Fx, Fy, sigma, na, lambda_)
    P_grid = pupil_response_function(Fx, Fy, na, lambda_)

    # 计算TCC
    for i_prime in range(Lx):  # fx'
        for j_prime in range(Ly):  # fy'
            for i_dprime in range(Lx):  # fx''
                for j_dprime in range(Ly):  # fy''

                    # 创建被积函数
                    integrand = np.zeros((Lx, Ly), dtype=complex)
                    for m in range(Lx):
                        for n in range(Ly):
                            # 计算P(f+f', g+g')
                            idx1_x = (m + i_prime) % Lx
                            idx1_y = (n + j_prime) % Ly
                            P_original = P_grid[idx1_x, idx1_y]

                            # 计算P*(f+f'', g+g'')
                            idx2_x = (m + i_dprime) % Lx
                            idx2_y = (n + j_dprime) % Ly
                            P_conj = np.conj(P_grid[idx2_x, idx2_y])

                            integrand[m, n] = J_grid[m, n] * P_original * P_conj
                            print("TCC...")

                    # 二重积分
                    integral_y = np.zeros(Lx, dtype=complex)
                    for i in range(Lx):
                        integral_y[i] = simpson(integrand[i, :], fy)
                    tcc_value = simpson(integral_y, fx)

                    tcc[i_prime, j_prime, i_dprime, j_dprime] = tcc_value

    return tcc


def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, na=NA, sigma=SIGMA,
                               Wx=WX, Wy=WY, Tx=TX, Ty=TY, dx=DX, dy=DY):
    """
    Hopkins成像仿真（基于四维TCC）
    """
    # 1. DMD调制
    M_prime = dmd_modulation(mask, Wx, Wy, Tx, Ty, dx, dy)

    # 2. 计算空间频率坐标
    Lx, Ly = M_prime.shape
    fx = np.fft.fftfreq(Lx, dx)
    fy = np.fft.fftfreq(Ly, dy)

    # 3. 计算TCC
    TCC = compute_tcc_4d(fx, fy, sigma, na, lambda_)

    # 4. 计算掩模频谱
    M_fft = fftshift(fft2(M_prime))

    # 5. Hopkins成像计算
    # 初始空间频域表示
    I_mask = np.zeros_like(M_fft, dtype=complex)

    # 四维积分实现
    Lx, Ly = M_fft.shape
    for i_prime in range(Lx):  # fx'
        for j_prime in range(Ly):  # fy'
            for i_dprime in range(Lx):  # fx''
                for j_dprime in range(Ly):  # fy''
                    I_mask[i_prime, j_prime] += (TCC[i_prime, j_prime, i_dprime, j_dprime] *
                                                 M_fft[i_prime, j_prime] *
                                                 np.conj(M_fft[i_dprime, j_dprime]))
                    print("Hopkins...")
    I_xy = np.abs(ifft2(ifftshift(I_mask)))

    return I_xy

