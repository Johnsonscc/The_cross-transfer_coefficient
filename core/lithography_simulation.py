import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.integrate import simpson
from config.parameters import *


def dmd_modulation(M_xy, Wx=WX, Wy=WY, Tx=TX, Ty=TY, dx=DX, dy=DY):
    """
    DMD调制函数
    """
    M, N = M_xy.shape

    # 1. 梳状函数 comb(x/Tx, y/Ty)
    samples_x = int(np.round(Tx / dx))
    samples_y = int(np.round(Ty / dy))

    comb_x = np.zeros(samples_x)
    comb_y = np.zeros(samples_y)

    # 在周期位置设置脉冲
    comb_x[0] = 1
    comb_y[0] = 1

    # 构建二维梳状函数
    comb_xy = np.outer(comb_x, comb_y)

    # 2. 矩形函数 rect(x/Wx, y/Wy)
    rect_samples_x = int(np.round(Wx / dx))
    rect_samples_y = int(np.round(Wy / dy))

    rect_x = np.zeros(samples_x)
    rect_y = np.zeros(samples_y)

    # 在中心位置设置矩形窗口
    start_x = (samples_x - rect_samples_x) // 2
    end_x = start_x + rect_samples_x
    start_y = (samples_y - rect_samples_y) // 2
    end_y = start_y + rect_samples_y

    rect_x[start_x:end_x] = 1
    rect_y[start_y:end_y] = 1

    # 构建二维矩形函数
    rect_xy = np.outer(rect_x, rect_y)

    # 3. 构建调制函数 (1/Tx*Ty) * [rect * comb]
    modulated_rc = (1 / (Tx * Ty)) * np.convolve(rect_xy.flatten(), comb_xy.flatten(), mode='same').reshape(
        rect_xy.shape)

    # 4. 掩模M(x,y)调制
    expand_M = np.zeros((M * samples_x, N * samples_y))

    for i in range(M):
        for j in range(N):
            start_row = i * samples_x
            start_col = j * samples_y
            end_row = (i + 1) * samples_x
            end_col = (j + 1) * samples_y

            # 调制窗口
            expand_M[start_row:end_row, start_col:end_col] = M_xy[i, j] * modulated_rc

    return expand_M


def compute_frequency(dx=DX, dy=DY, lx=LX, ly=LY):
    """
    计算频率坐标
    """
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), lx)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), ly)

    Fx, Fy = np.meshgrid(fx, fy, indexing='ij')
    return Fx, Fy


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
    P = np.where(r <= r_max, 1.0, 0)  # 简化为单位振幅
    return P


def compute_tcc_full(fx, fy, J, P, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    """
    完整的四维TCC计算 - 按照伪代码实现
    """
    Lx, Ly = fx.shape

    # 初始化四维TCC矩阵
    tcc = np.zeros((Lx, Ly, Lx, Ly), dtype=complex)

    # 预计算光源函数网格
    J_grid = light_source_function(fx, fy, sigma, na, lambda_)

    # 计算TCC - 四重循环（注意：计算量很大，适合小尺寸）
    for i_prime in range(Lx):  # fx'
        for j_prime in range(Ly):  # fy'
            for i_dprime in range(Lx):  # fx''
                for j_dprime in range(Ly):  # fy''

                    # 创建被积函数
                    integrand = np.zeros((Lx, Ly), dtype=complex)
                    print("Tcc ...")
                    for m in range(Lx):
                        for n in range(Ly):
                            # 计算P(f+f', g+g')
                            P_original = pupil_response_function(
                                fx[m, n] + fx[i_prime, j_prime],
                                fy[m, n] + fy[i_prime, j_prime],
                                na, lambda_
                            )

                            # 计算P*(f+f'', g+g'')
                            P_conj = np.conj(pupil_response_function(
                                fx[m, n] + fx[i_dprime, j_dprime],
                                fy[m, n] + fy[i_dprime, j_dprime],
                                na, lambda_
                            ))

                            integrand[m, n] = J_grid[m, n] * P_original * P_conj

                    # 二重积分
                    integral_y = np.zeros(Lx, dtype=complex)
                    for i in range(Lx):
                        integral_y[i] = simpson(integrand[i, :], fy[i, :])

                    tcc_value = simpson(integral_y, fx[:, 0])
                    tcc[i_prime, j_prime, i_dprime, j_dprime] = tcc_value

    return tcc


def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, na=NA, sigma=SIGMA,
                               Wx=WX, Wy=WY, Tx=TX, Ty=TY, dx=DX, dy=DY):
    """
    Hopkins成像模拟 - 按照伪代码实现完整流程
    """
    # 1. DMD调制
    M_prime = dmd_modulation(mask, Wx, Wy, Tx, Ty, dx, dy)

    # 2. 计算空间频率坐标
    Lx, Ly = M_prime.shape
    fx = np.fft.fftfreq(Lx, dx)
    fy = np.fft.fftfreq(Ly, dy)
    Fx, Fy = np.meshgrid(fx, fy, indexing='ij')

    # 3. 计算光源函数
    J = light_source_function(Fx, Fy, sigma, na, lambda_)

    # 4. 计算光瞳响应函数
    P = pupil_response_function(Fx, Fy, na, lambda_)

    # 5. 计算TCC（简化版，实际应用可能需要优化）
    # 注意：完整TCC计算非常耗时，对于大尺寸图像需要优化
    TCC = compute_tcc_full(Fx, Fy, J, P, sigma, na, lambda_)

    # 6. 计算掩模频谱
    M_fft = fftshift(fft2(M_prime))

    # 7. Hopkins成像计算
    I_mask = np.zeros_like(M_fft, dtype=complex)

    # 四维积分实现
    for i_prime in range(Lx):  # fx'
        for j_prime in range(Ly):  # fy'
            for i_dprime in range(Lx):  # fx''
                for j_dprime in range(Ly):  # fy''
                    I_mask[i_prime, j_prime] += (
                            TCC[i_prime, j_prime, i_dprime, j_dprime] *
                            M_fft[i_prime, j_prime] *
                            np.conj(M_fft[i_dprime, j_dprime])
                    )
                    print("simpon ...")
    I_xy = np.abs(ifft2(ifftshift(I_mask)))

    return I_xy




