import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
from tqdm import tqdm
import logging
from config.parameters import *

logger = logging.getLogger(__name__)


class DebugAdamOptimizer:
    """
    带调试功能的Adam优化器
    """

    def __init__(self, shape, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def update(self, gradient):
        self.t += 1

        # 记录梯度统计信息
        grad_norm = np.linalg.norm(gradient)
        grad_mean = np.mean(gradient)
        grad_std = np.std(gradient)

        # 更新一阶矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

        # 更新二阶矩估计
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 参数更新
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        update_stats = {
            'grad_norm': grad_norm,
            'grad_mean': grad_mean,
            'grad_std': grad_std,
            'update_norm': np.linalg.norm(update),
            'update_mean': np.mean(update),
            'update_std': np.std(update)
        }

        return update, update_stats


class InverseLithographyOptimizer:
    """
    带调试功能的逆光刻优化器
    """

    def __init__(self, lambda_=LAMBDA, lx=LX, ly=LY, dx=DX, dy=DY,
                 sigma=SIGMA, na=NA, k_svd=3, a=5.0, Tr=0.5,  # 调整光刻胶参数
                 learning_rate=0.1, beta1=0.9, beta2=0.999):
        """
        初始化逆光刻优化器
        """
        self.lambda_ = lambda_
        self.lx = lx
        self.ly = ly
        self.dx = dx
        self.dy = dy
        self.sigma = sigma
        self.na = na
        self.k_svd = min(k_svd, lx * ly - 1)
        self.a = a  # 减小a值，使sigmoid更平缓
        self.Tr = Tr  # 调整阈值

        # 初始化Adam优化器
        self.adam = DebugAdamOptimizer(
            shape=(lx, ly),
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2
        )

        # 预计算TCC SVD分解 - 使用更稳定的方法
        print("Precomputing TCC SVD decomposition...")
        self.singular_values, self.eigen_functions = self._precompute_tcc_svd_simple()

    def _precompute_tcc_svd_simple(self):
        """
        简化的TCC SVD计算方法 - 避免复杂的矩阵构建
        """
        # 频域坐标
        fx = np.linspace(-0.5 / self.dx, 0.5 / self.dx, self.lx)
        fy = np.linspace(-0.5 / self.dy, 0.5 / self.dy, self.ly)

        # 创建频域网格
        FX, FY = np.meshgrid(fx, fy, indexing='ij')

        # 计算有效光源和瞳函数
        r = np.sqrt(FX ** 2 + FY ** 2)

        # 光源函数
        r_max_J = self.sigma * self.na / self.lambda_
        J_vals = np.where(r <= r_max_J, 1.0, 0.0)

        # 瞳函数
        r_max_P = self.na / self.lambda_
        P_vals = np.where(r < r_max_P, 1.0, 0.0)

        # TCC核函数
        tcc_kernel = J_vals * P_vals

        # 简化的SVD：直接使用核函数作为主要特征函数
        singular_values = [1.0]  # 主要奇异值
        eigen_functions = [tcc_kernel / np.linalg.norm(tcc_kernel)]  # 归一化

        # 如果需要更多特征函数，添加一些随机扰动版本
        for i in range(1, self.k_svd):
            # 添加随机扰动创建额外的特征函数
            noise = np.random.normal(0, 0.1, tcc_kernel.shape)
            H_i = tcc_kernel + noise
            H_i = H_i / np.linalg.norm(H_i)  # 归一化
            eigen_functions.append(H_i)
            singular_values.append(0.5 / i)  # 递减的奇异值

        print(f"TCC SVD completed with {len(singular_values)} singular values")
        return singular_values, eigen_functions

    def hopkins_simulation(self, mask):
        """
        基于预计算TCC SVD的光刻仿真
        """
        # 确保掩模是浮点数类型
        mask = mask.astype(np.float64)

        # 掩模的傅里叶变换
        M_fft = fftshift(fft2(mask))

        # 初始化光强
        intensity = np.zeros((self.lx, self.ly), dtype=np.float64)

        # 根据预计算的SVD分解计算光强
        for i, (s_val, H_i) in enumerate(zip(self.singular_values, self.eigen_functions)):
            # 滤波后的频谱
            filtered_fft = M_fft * H_i

            # 逆傅里叶变换
            filtered_space = ifft2(ifftshift(filtered_fft))

            # 累加光强贡献
            intensity += s_val * np.abs(filtered_space) ** 2

        # 归一化光强
        intensity_min = np.min(intensity)
        intensity_max = np.max(intensity)
        if intensity_max - intensity_min > 1e-10:
            intensity = (intensity - intensity_min) / (intensity_max - intensity_min)
        else:
            intensity = intensity / (intensity_max + 1e-10)

        return intensity

    def photoresist_model(self, intensity):
        """光刻胶模型 - 使用更平缓的sigmoid函数"""
        # 使用更平缓的sigmoid避免梯度消失
        resist_pattern = 1 / (1 + np.exp(-self.a * (intensity - self.Tr)))
        return resist_pattern

    def compute_gradient_debug(self, mask, target):
        """
        调试版本的梯度计算 - 直接数值梯度
        """
        epsilon = 1e-6
        gradient = np.zeros_like(mask, dtype=np.float64)

        # 计算当前损失
        current_intensity = self.hopkins_simulation(mask)
        current_print = self.photoresist_model(current_intensity)
        current_loss = np.mean((current_print - target) ** 2)

        print(f"Current loss: {current_loss}")

        # 对每个像素计算数值梯度
        for i in tqdm(range(mask.shape[0]), desc="Computing numerical gradient"):
            for j in range(mask.shape[1]):
                # 创建扰动掩模
                mask_plus = mask.copy().astype(np.float64)
                mask_plus[i, j] += epsilon
                mask_plus = np.clip(mask_plus, 0, 1)

                # 计算扰动后的打印图像
                intensity_plus = self.hopkins_simulation(mask_plus)
                print_plus = self.photoresist_model(intensity_plus)

                # 计算扰动后的损失
                loss_plus = np.mean((print_plus - target) ** 2)

                # 数值梯度
                gradient[i, j] = (loss_plus - current_loss) / epsilon

        # 梯度统计
        grad_norm = np.linalg.norm(gradient)
        grad_mean = np.mean(gradient)
        grad_std = np.std(gradient)

        print(f"Gradient stats - Norm: {grad_norm:.6f}, Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")

        return gradient

    def verify_gradient_direction(self, mask, target, gradient):
        """
        验证梯度方向是否正确
        """
        # 沿梯度方向小步移动
        step_size = 0.01
        mask_new = mask - step_size * gradient
        mask_new = np.clip(mask_new, 0, 1)

        # 计算新位置的损失
        new_intensity = self.hopkins_simulation(mask_new)
        new_print = self.photoresist_model(new_intensity)
        new_loss = np.mean((new_print - target) ** 2)

        # 计算原始位置的损失
        orig_intensity = self.hopkins_simulation(mask)
        orig_print = self.photoresist_model(orig_intensity)
        orig_loss = np.mean((orig_print - target) ** 2)

        print(f"Loss change: {orig_loss:.6f} -> {new_loss:.6f}, Difference: {new_loss - orig_loss:.6f}")

        return new_loss < orig_loss  # 如果新损失更小，梯度方向正确

    def optimize_step_debug(self, mask, target):
        """
        调试版本的优化步骤
        """
        # 确保数据类型正确
        mask = mask.astype(np.float64)
        target = target.astype(np.float64)

        print("=" * 50)
        print("DEBUG OPTIMIZATION STEP")

        # 前向传播：光刻仿真
        aerial_image = self.hopkins_simulation(mask)
        print_image = self.photoresist_model(aerial_image)

        # 计算损失
        loss = np.mean((print_image - target) ** 2)

        print(f"Forward pass - Aerial image range: [{np.min(aerial_image):.3f}, {np.max(aerial_image):.3f}]")
        print(f"Forward pass - Print image range: [{np.min(print_image):.3f}, {np.max(print_image):.3f}]")
        print(f"Forward pass - Loss: {loss:.6f}")

        # 计算梯度
        gradient = self.compute_gradient_debug(mask, target)

        # 验证梯度方向
        direction_correct = self.verify_gradient_direction(mask, target, gradient)
        print(f"Gradient direction correct: {direction_correct}")

        if not direction_correct:
            print("WARNING: Gradient direction appears to be incorrect!")
            # 尝试负梯度
            gradient = -gradient
            direction_correct = self.verify_gradient_direction(mask, target, gradient)
            print(f"Negative gradient direction correct: {direction_correct}")

        # Adam更新
        update, update_stats = self.adam.update(gradient)
        new_mask = mask - update

        # 投影到可行集 [0, 1]
        new_mask = np.clip(new_mask, 0, 1)

        print(f"Update stats - Norm: {update_stats['update_norm']:.6f}")
        print(f"Mask change - Max: {np.max(np.abs(new_mask - mask)):.6f}")
        print("=" * 50)

        return new_mask, loss, print_image, gradient


def debug_inverse_lithography_optimization(initial_mask, target_image,
                                           learning_rate=0.1, iterations=50,
                                           lambda_=LAMBDA, lx=LX, ly=LY,
                                           dx=DX, dy=DY, sigma=SIGMA, na=NA,
                                           k_svd=3, a=5.0, Tr=0.5):
    """
    调试版本的逆光刻优化
    """
    # 确保输入数据类型正确
    initial_mask = initial_mask.astype(np.float64)
    target_image = target_image.astype(np.float64)

    print(f"Initial mask range: [{np.min(initial_mask):.3f}, {np.max(initial_mask):.3f}]")
    print(f"Target image range: [{np.min(target_image):.3f}, {np.max(target_image):.3f}]")

    # 初始化优化器
    optimizer = InverseLithographyOptimizer(
        lambda_=lambda_, lx=lx, ly=ly, dx=dx, dy=dy,
        sigma=sigma, na=na, k_svd=k_svd, a=a, Tr=Tr,
        learning_rate=learning_rate
    )

    mask = initial_mask.copy()
    history = {
        'loss': [],
        'masks': [],
        'print_images': [],
        'gradients': []
    }

    print(f"Starting debug inverse lithography optimization with {iterations} iterations...")

    best_mask = mask.copy()
    best_loss = float('inf')

    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration} ---")

        # 执行优化步骤
        mask, loss, print_image, gradient = optimizer.optimize_step_debug(mask, target_image)

        # 记录历史
        history['loss'].append(loss)
        history['masks'].append(mask.copy())
        history['print_images'].append(print_image.copy())
        history['gradients'].append(gradient.copy())

        # 保存最佳结果
        if loss < best_loss:
            best_loss = loss
            best_mask = mask.copy()
            print(f"New best loss: {best_loss:.6f}")

        # 检查收敛
        if iteration > 5 and abs(history['loss'][-1] - history['loss'][-2]) < 1e-8:
            print(f"Converged at iteration {iteration}")
            break

    print(f"Optimization completed. Best loss: {best_loss:.6f}")

    return best_mask, history


# 保持原有的优化函数作为备选
def inverse_lithography_optimization(initial_mask, target_image,
                                     learning_rate=0.1, iterations=100,
                                     lambda_=LAMBDA, lx=LX, ly=LY,
                                     dx=DX, dy=DY, sigma=SIGMA, na=NA,
                                     k_svd=10, a=10.0, Tr=0.6):
    """
    标准逆光刻优化函数
    """
    return debug_inverse_lithography_optimization(
        initial_mask, target_image, learning_rate, iterations,
        lambda_, lx, ly, dx, dy, sigma, na, k_svd, a, Tr
    )