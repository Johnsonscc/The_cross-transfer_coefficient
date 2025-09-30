import numpy as np
from scipy.signal import convolve2d


def dmd_modulation(M_xy, Lx, Ly, Wx, Wy, Tx, Ty, dx, dy):
    # 计算超采样因子以获得更精确的卷积
    oversample_factor = 10

    # 超采样输入图像
    M_xy= np.kron(M_xy, np.ones((oversample_factor, oversample_factor)))

    # 创建空间坐标网格
    x = np.arange(-Lx // 2, Lx // 2) * dx
    y = np.arange(-Ly // 2, Ly // 2) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 1. 矩形函数 rect(x/Wx, y/Wy)
    rect_function = np.zeros((Lx, Ly))
    rect_function[(np.abs(X) <= Wx / 2) & (np.abs(Y) <= Wy / 2)] = 1.0

    # 2. 梳状函数 comb(x/Tx, y/Ty) - 用狄拉克δ函数的离散近似
    comb_function = np.zeros((Lx, Ly))
    # 在周期位置设置脉冲
    for i in range(Lx):
        for j in range(Ly):
            if (abs(X[i, j] % Tx) < dx / 2) and (abs(Y[i, j] % Ty) < dy / 2):
                comb_function[i, j] = 1.0 / (dx * dy)  # 狄拉克δ函数的离散近似

    # 3. 矩形函数与梳状函数的卷积
    # 使用FFT卷积提高效率
    rect_comb_conv = convolve2d(rect_function, comb_function, mode='same', boundary='fill', fillvalue=0)

    # 4. 原始掩模与调制函数的卷积，并应用归一化因子
    M_prime = convolve2d(M_xy, rect_comb_conv, mode='same', boundary='fill', fillvalue=0)
    M_prime = M_prime / (Tx * Ty)

    # 确保输出在合理范围内
    M_prime = np.clip(M_prime, 0, 1)

    return M_prime