import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from config.parameters import *
from scipy.signal import convolve2d

#DMD调制
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
    rect_comb_conv = convolve2d(rect_function, comb_function, mode='same', boundary='fill', fillvalue=0)

    # 4. 原始掩模与调制函数的卷积，并应用归一化因子
    M_prime = convolve2d(M_xy, rect_comb_conv, mode='same', boundary='fill', fillvalue=0)
    M_prime = M_prime / (Tx * Ty)

    # 确保输出在合理范围内
    M_prime = np.clip(M_prime, 0, 1)

    return M_prime



def transfer_function(fx, fy, lambda_=LAMBDA, z=Z, n=N):
    H = np.exp(-1j * np.pi * lambda_ * z * (fx ** 2 + fy ** 2) / n ** 2)#与原文中的霍普金斯存在偏差
    return H

#曝光光源强度函数
def light_source_function(fx,fy,sigma=SIGMA,na=NA,lambda_=LAMBDA):
	r=np.sqrt(fx**2+fy**2)
	r_max=sigma*na/lambda_
	J=np.where(r<=r_max,lambda_**2/(np.pi*(sigma*na)**2),0)
	return J

#光瞳频率响应函数
def pupil_response_function(fx,fy,na=NA,lambda_=LAMBDA):
	r=np.sqrt(fx**2+fy**2)
	r_max=na/lambda_
	P=np.where(r<=r_max,lambda_**2/(np.pi*(na)**2),0)
	return P

def compute_tcc(J, P, fx, fy):
    tcc = np.convolve(J(fx, fy) * P(fx, fy), J(fx, fy) * P(fx, fy), mode='same')
    return tcc


def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                           wx=WX,wy=WY,tx=TX,ty=TY,z=Z, dx=DX, dy=DY, n=N, sigma=SIGMA, na=NA):

    #mask = dmd_modulation(mask, lx, ly, wx, wy, tx, ty, dx, dy)


    fx = np.linspace(-0.5 / dx, 0.5 / dx, lx)
    fy = np.linspace(-0.5 / dy, 0.5 / dy, ly)

    J = lambda fx, fy: light_source_function(fx, fy, sigma, na, lambda_)
    P = lambda fx, fy: pupil_response_function(fx, fy, na, lambda_)

    tcc = compute_tcc(J, P, fx, fy)
    M_fft = fftshift(fft2(mask))
    filtered_fft = M_fft * tcc

    H = transfer_function(fx, fy, lambda_, z, n)
    result_fft = filtered_fft * H

    result = ifft2(ifftshift(result_fft))
    return np.abs(result)

