import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from config.parameters import *
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_gradient(mask, target_image, singular_values, kernel_functions, resist_a, resist_Tr):
    """
    计算评价函数对掩模的梯度

    参数:
    mask: 当前掩模
    target_image: 目标图像
    singular_values: 奇异值
    kernel_functions: 核函数
    resist_a: 光刻胶参数a
    resist_Tr: 光刻胶参数Tr

    返回:
    gradient: 梯度
    resist_output: 光刻胶输出
    intensity: 光强分布
    loss: 当前损失值
    """
    # 光刻仿真
    mask_fft = fftshift(fft2(mask))
    intensity = np.zeros_like(mask, dtype=np.float64)

    for i, (s_val, H_i) in enumerate(zip(singular_values, kernel_functions)):
        filtered_fft = mask_fft * H_i
        filtered_space = ifft2(ifftshift(filtered_fft))
        intensity += s_val * np.abs(filtered_space) ** 2

    # 归一化
    intensity_normalized = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

    # 光刻胶模型
    resist_output = 1 / (1 + np.exp(-resist_a * (intensity_normalized - resist_Tr)))

    # 计算损失
    loss = np.sum((resist_output - target_image) ** 2)

    # 计算梯度
    dF_dz = 2 * (resist_output - target_image)
    dz_dI = resist_a * resist_output * (1 - resist_output)
    dF_dI = dF_dz * dz_dI

    gradient = np.zeros_like(mask)

    for i, (s_val, H_i) in enumerate(zip(singular_values, kernel_functions)):
        # h_i ⊗ ω
        hi_omega_fft = mask_fft * H_i
        hi_omega = ifft2(ifftshift(hi_omega_fft))

        # h_i^* ⊗ (h_i ⊗ ω)
        hi_conj = np.conj(H_i)
        temp_fft = fftshift(fft2(hi_omega)) * hi_conj
        hi_conj_hi_omega = ifft2(ifftshift(temp_fft))

        # 实部并加权
        grad_component = 2 * s_val * np.real(hi_conj_hi_omega)
        gradient += dF_dI * grad_component

    return gradient, resist_output, intensity_normalized, loss


def inverse_lithography_optimization(initial_mask, target_image,
                                     learning_rate=ILT_LEARNING_RATE,
                                     max_iter=ILT_MAX_ITER,
                                     resist_a=A,
                                     resist_Tr=TR,
                                     convergence_tol=ILT_CONVERGENCE_TOL,
                                     k_svd=ILT_SVD_K):
    """
    逆光刻优化主函数

    参数:
    initial_mask: 初始掩模
    target_image: 目标图像
    learning_rate: 学习率
    max_iter: 最大迭代次数
    resist_a: 光刻胶参数a
    resist_Tr: 光刻胶参数Tr
    convergence_tol: 收敛容差
    k_svd: SVD分解保留的奇异值数量

    返回:
    optimized_mask: 优化后的掩模
    optimization_history: 优化历史记录
    """
    from core.lithography_simulation import compute_tcc_svd, light_source_function, pupil_response_function

    # 获取图像尺寸
    lx, ly = initial_mask.shape

    # 计算频域坐标
    dx, dy = DX, DY
    fx = np.linspace(-0.5 / dx, 0.5 / dx, lx)
    fy = np.linspace(-0.5 / dy, 0.5 / dy, ly)

    # 定义光源和瞳函数
    J = lambda fx, fy: light_source_function(fx, fy, SIGMA, NA, LAMBDA)
    P = lambda fx, fy: pupil_response_function(fx, fy, NA, LAMBDA)

    # 计算TCC并进行SVD分解
    print("Computing TCC SVD decomposition...")
    singular_values, kernel_functions = compute_tcc_svd(J, P, fx, fy, k_svd)

    # 初始化
    current_mask = initial_mask.copy()
    optimization_history = {
        'masks': [],
        'losses': [],
        'resist_outputs': [],
        'intensities': []
    }

    print("Starting ILT lithography optimization...")
    prev_loss = float('inf')

    for iteration in tqdm(range(max_iter), desc="ILT Lithography"):
        # 计算梯度和当前状态
        gradient, resist_output, intensity, loss = compute_gradient(
            current_mask, target_image, singular_values, kernel_functions, resist_a, resist_Tr
        )

        # 记录历史
        optimization_history['masks'].append(current_mask.copy())
        optimization_history['losses'].append(loss)
        optimization_history['resist_outputs'].append(resist_output.copy())
        optimization_history['intensities'].append(intensity.copy())

        # 检查收敛
        if abs(prev_loss - loss) < convergence_tol:
            print(f"Converged at iteration {iteration}, loss: {loss:.6f}")
            break

        prev_loss = loss

        # 更新掩模
        current_mask = current_mask - learning_rate * gradient

        # 投影到可行域 [0, 1]
        current_mask = np.clip(current_mask, 0, 1)

        # 打印进度
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss:.6f}")

    optimized_mask = current_mask

    return optimized_mask, optimization_history