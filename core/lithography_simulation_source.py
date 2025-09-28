import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from config.parameters import *


def transfer_function(fx, fy, lambda_=LAMBDA, z=Z, n=N):
    H = np.exp(-1j * np.pi * lambda_ * z * (fx ** 2 + fy ** 2) / n ** 2)#与原文中的霍普金斯存在偏差
    return H


def light_source_function(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    condition = (fx ** 2 + fy ** 2) <= (sigma * na / lambda_) ** 2
    J = np.where(condition, (lambda_ ** 2) / (np.pi * (sigma * na) ** 2), 0)
    return J


def impulse_response_function(fx, fy, na=NA, lambda_=LAMBDA):
    condition = (fx ** 2 + fy ** 2) <= (na / lambda_) ** 2
    P = np.where(condition, (lambda_ ** 2) / (np.pi * na ** 2), 0)
    return P


def compute_tcc(J, P, fx, fy):
    tcc = np.convolve(J(fx, fy) * P(fx, fy), J(fx, fy) * P(fx, fy), mode='same')
    return tcc


def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                           z=Z, dx=DX, dy=DY, n=N, sigma=SIGMA, na=NA):
    fx = np.linspace(-0.5 / dx, 0.5 / dx, lx)
    fy = np.linspace(-0.5 / dy, 0.5 / dy, ly)

    J = lambda fx, fy: light_source_function(fx, fy, sigma, na, lambda_)
    P = lambda fx, fy: impulse_response_function(fx, fy, na, lambda_)

    tcc = compute_tcc(J, P, fx, fy)
    M_fft = fftshift(fft2(mask))
    filtered_fft = M_fft * tcc

    H = transfer_function(fx, fy, lambda_, z, n)
    result_fft = filtered_fft * H

    result = ifft2(ifftshift(result_fft))
    return np.abs(result)