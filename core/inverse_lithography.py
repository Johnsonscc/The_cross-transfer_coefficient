import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
import logging
from tqdm import tqdm
from config.parameters import *

logger = logging.getLogger(__name__)


class InverseLithographyOptimizer:
    """
    逆光刻优化器 - 基于完整数学公式的解析梯度计算
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA, dx=DX, dy=DY,
                 lx=LX, ly=LY, k_svd=10, a=A, tr=TR):
        # 光学参数
        self.lambda_ = lambda_
        self.na = na
        self.sigma = sigma
        self.dx = dx
        self.dy = dy
        self.lx = lx
        self.ly = ly
        self.k_svd = k_svd

        # 光刻胶参数
        self.a = a
        self.tr = tr

        # 预计算TCC SVD分解
        self.singular_values = None
        self.eigen_functions = None
        self._precompute_tcc_svd()

        logger.info("InverseLithographyOptimizer initialized")

    def _precompute_tcc_svd(self):
        """预计算TCC的SVD分解"""
        print("Precomputing TCC SVD decomposition...")

        # 频域坐标
        fx = np.linspace(-0.5 / self.dx, 0.5 / self.dx, self.lx)
        fy = np.linspace(-0.5 / self.dy, 0.5 / self.dy, self.ly)

        # 完整计算TCC矩阵
        TCC_4d = self._compute_full_tcc_matrix(fx, fy)

        # 对TCC矩阵进行SVD分解
        self.singular_values, self.eigen_functions = self._svd_of_tcc_matrix(TCC_4d, self.k_svd)

        print(f"TCC SVD precomputation completed with {len(self.singular_values)} singular values")

    def _compute_full_tcc_matrix(self, fx, fy):
        """计算完整的4D TCC矩阵"""
        Lx, Ly = len(fx), len(fy)

        # 创建频域网格
        FX, FY = np.meshgrid(fx, fy, indexing='ij')

        # 计算光源函数 J(f,g)
        r_j = np.sqrt(FX ** 2 + FY ** 2)
        r_max_j = self.sigma * self.na / self.lambda_
        J = np.where(r_j <= r_max_j, 1.0, 0.0)
        J = J / np.sum(J)  # 归一化

        # 计算瞳函数 P(f,g)
        r_p = np.sqrt(FX ** 2 + FY ** 2)
        r_max_p = self.na / self.lambda_
        P = np.where(r_p <= r_max_p, 1.0, 0.0).astype(np.complex128)

        tcc_kernel = J * P

        print(f"Building 4D TCC matrix ({Lx}x{Ly}x{Lx}x{Ly})...")

        # 初始化4D TCC矩阵
        TCC_4d = np.zeros((Lx, Ly, Lx, Ly), dtype=np.complex128)

        # TCC计算
        for i in tqdm(range(Lx), desc="TCC Computation"):
            for j in range(Ly):
                for m in range(Lx):
                    for n in range(Ly):
                        if (0 <= i - m < Lx) and (0 <= j - n < Ly):
                            TCC_4d[i, j, m, n] = tcc_kernel[i, j] * np.conj(tcc_kernel[m, n])

        return TCC_4d

    def _svd_of_tcc_matrix(self, TCC_4d, k):
        """对4D TCC矩阵进行SVD分解"""
        Lx, Ly, _, _ = TCC_4d.shape

        print("Reshaping TCC matrix for SVD...")
        TCC_2d = TCC_4d.reshape(Lx * Ly, Lx * Ly)

        print("Performing SVD decomposition...")
        k_actual = min(k, min(TCC_2d.shape) - 1)
        U, s, Vh = svds(TCC_2d, k=k_actual)

        # 确保奇异值按降序排列
        idx = np.argsort(s)[::-1]
        s = s[idx]
        U = U[:, idx]

        H_functions = []
        for i in tqdm(range(len(s)), desc="Extracting eigenfunctions"):
            H_i = U[:, i].reshape(Lx, Ly)
            H_functions.append(H_i)

        return s, H_functions

    def hopkins_simulation(self, mask):
        """Hopkins光刻仿真"""
        M_fft = fftshift(fft2(mask))
        intensity = np.zeros((self.lx, self.ly), dtype=np.float64)

        for i, (s_val, H_i) in enumerate(zip(self.singular_values, self.eigen_functions)):
            filtered_fft = M_fft * H_i
            filtered_space = ifft2(ifftshift(filtered_fft))
            intensity += s_val * np.abs(filtered_space) ** 2

        # 归一化
        intensity_min = np.min(intensity)
        intensity_max = np.max(intensity)

        if intensity_max - intensity_min > 1e-10:
            result = (intensity - intensity_min) / (intensity_max - intensity_min)
        else:
            result = intensity / (intensity_max + 1e-10)

        return result

    def photoresist_model(self, intensity):
        """光刻胶模型 - sigmoid函数"""
        return 1 / (1 + np.exp(-self.a * (intensity - self.tr)))

    def _compute_analytical_gradient(self, mask, target):
        """
        基于完整数学公式的解析梯度计算

        根据公式:
        F(ω) = ∑ [z_ξ - 1/(1+exp(-a(∑σ_i|h_i⊗M|² - T_r)))]²

        计算梯度: ∂F/∂M
        """
        # 前向传播计算中间变量
        M_fft = fftshift(fft2(mask))

        # 存储中间结果用于梯度计算
        A_i_list = []  # A_i = h_i ⊗ M
        I_i_list = []  # I_i = |A_i|²

        # 计算光强和各中间项
        intensity = np.zeros((self.lx, self.ly), dtype=np.float64)
        for i, (s_val, H_i) in enumerate(zip(self.singular_values, self.eigen_functions)):
            A_i_fft = M_fft * H_i
            A_i = ifft2(ifftshift(A_i_fft))
            I_i = np.abs(A_i) ** 2

            A_i_list.append(A_i)
            I_i_list.append(I_i)
            intensity += s_val * I_i

        # 归一化光强
        intensity_min = np.min(intensity)
        intensity_max = np.max(intensity)
        if intensity_max - intensity_min > 1e-10:
            intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)
        else:
            intensity_norm = intensity / (intensity_max + 1e-10)

        # 光刻胶输出
        P = self.photoresist_model(intensity_norm)

        # 计算损失
        loss = np.sum((target - P) ** 2)

        # 计算梯度 ∂F/∂M
        gradient = np.zeros_like(mask, dtype=np.complex128)

        # 链式法则: ∂F/∂M = ∂F/∂P * ∂P/∂I * ∂I/∂M

        # 1. ∂F/∂P = -2(z - P)
        dF_dP = -2 * (target - P)

        # 2. ∂P/∂I = a * P * (1 - P) * (∂I_norm/∂I)
        # 注意这里要考虑归一化的影响
        dP_dI_norm = self.a * P * (1 - P)

        # 归一化的导数
        if intensity_max - intensity_min > 1e-10:
            dI_norm_dI = 1.0 / (intensity_max - intensity_min)
            # 减去均值的影响
            dI_norm_dI -= (intensity - intensity_min) / (intensity_max - intensity_min) ** 2 * (
                    (intensity == intensity_max).astype(float) - (intensity == intensity_min).astype(float)
            )
        else:
            dI_norm_dI = 1.0 / (intensity_max + 1e-10)

        dP_dI = dP_dI_norm * dI_norm_dI

        # 3. ∂I/∂M 的计算
        for i, (s_val, H_i, A_i, I_i) in enumerate(zip(
                self.singular_values, self.eigen_functions, A_i_list, I_i_list
        )):
            # ∂I/∂M = 2 * σ_i * Re{A_i * (∂(h_i⊗M)/∂M)}
            # 由于 h_i⊗M 是线性操作，其导数是 h_i 的共轭

            # 计算 ∂F/∂A_i = ∂F/∂I * ∂I/∂A_i = dF_dP * dP_dI * 2 * σ_i * A_i
            dF_dA_i = dF_dP * dP_dI * 2 * s_val * A_i.conj()

            # 通过傅里叶变换计算梯度贡献
            dF_dA_i_fft = fftshift(fft2(dF_dA_i))
            gradient_contribution = ifft2(ifftshift(dF_dA_i_fft * np.conj(H_i)))

            gradient += gradient_contribution

        # 取实部，因为掩模是实数
        gradient_real = np.real(gradient)

        return loss, gradient_real, intensity_norm, P

    def compute_loss_and_gradient(self, mask, target):
        """
        统一的损失和梯度计算
        使用解析梯度方法
        """
        return self._compute_analytical_gradient(mask, target)

    def verify_gradient_analytical(self, mask, target, epsilon=1e-6):
        """
        验证解析梯度的正确性
        通过数值梯度进行验证
        """
        print("Verifying analytical gradient...")

        # 解析梯度
        analytical_loss, analytical_grad, _, _ = self._compute_analytical_gradient(mask, target)

        # 数值梯度
        numerical_grad = np.zeros_like(mask)
        base_loss, _, _, _ = self._compute_analytical_gradient(mask, target)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask_plus = mask.copy()
                mask_plus[i, j] += epsilon
                loss_plus, _, _, _ = self._compute_analytical_gradient(mask_plus, target)

                numerical_grad[i, j] = (loss_plus - base_loss) / epsilon

        # 比较
        diff = np.abs(analytical_grad - numerical_grad)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)

        print(f"Gradient verification - Max diff: {max_diff:.6f}, Avg diff: {avg_diff:.6f}")
        print(f"Analytical loss: {analytical_loss:.6f}, Base loss: {base_loss:.6f}")

        if max_diff < 1e-4:
            print("Analytical gradient verified successfully!")
            return True
        else:
            print("WARNING: Large discrepancy in gradient calculation!")
            return False

    def optimize(self, initial_mask, target, learning_rate=0.1, max_iterations=100):
        """
        使用解析梯度的优化过程
        """
        mask = initial_mask.copy()
        history = {
            'loss': [],
            'masks': [],
            'aerial_images': [],
            'printed_images': []
        }

        print(f"Starting ILT optimization with {max_iterations} iterations...")
        print("Using analytical gradient computation...")

        # 梯度验证
        self.verify_gradient_analytical(mask, target)

        best_mask = mask.copy()
        best_loss = float('inf')

        for iteration in tqdm(range(max_iterations), desc="ILT Optimization"):
            # 使用解析梯度计算损失和梯度
            loss, gradient, aerial_image, printed_image = self.compute_loss_and_gradient(mask, target)

            # 记录最佳掩模
            if loss < best_loss:
                best_loss = loss
                best_mask = mask.copy()

            # 梯度下降更新
            mask = mask - learning_rate * gradient
            mask = np.clip(mask, 0, 1)  # 投影到可行集

            # 记录历史
            history['loss'].append(loss)
            if iteration % 20 == 0 or iteration == max_iterations - 1:
                history['masks'].append(mask.copy())
                history['aerial_images'].append(aerial_image)
                history['printed_images'].append(printed_image)

            # 打印进度
            if iteration % 10 == 0:
                grad_norm = np.linalg.norm(gradient)
                print(f"Iteration {iteration}: Loss = {loss:.6f}, Grad Norm = {grad_norm:.6f}")

            # 早期停止
            if loss < 1e-6 and iteration > 20:
                print(f"Early stopping at iteration {iteration}")
                break

        print(f"Optimization completed. Best loss: {best_loss:.6f}")
        return best_mask, history

    def analyze_optimization_result(self, initial_mask, optimized_mask, target):
        """
        分析优化结果
        """
        print("\n=== Optimization Analysis ===")

        # 初始状态
        aerial_initial = self.hopkins_simulation(initial_mask)
        printed_initial = self.photoresist_model(aerial_initial)
        loss_initial, _, _, _ = self.compute_loss_and_gradient(initial_mask, target)

        # 优化后状态
        aerial_optimized = self.hopkins_simulation(optimized_mask)
        printed_optimized = self.photoresist_model(aerial_optimized)
        loss_optimized, _, _, _ = self.compute_loss_and_gradient(optimized_mask, target)

        # 掩模变化分析
        mask_change = np.abs(optimized_mask - initial_mask)

        # 辅助图形统计
        assist_features = np.sum((optimized_mask > 0.2) & (optimized_mask < 0.8))

        print(f"Initial loss: {loss_initial:.6f}")
        print(f"Optimized loss: {loss_optimized:.6f}")
        print(
            f"Improvement: {loss_initial - loss_optimized:.6f} ({((loss_initial - loss_optimized) / loss_initial * 100):.2f}%)")
        print(f"Mask change - Max: {np.max(mask_change):.3f}, Mean: {np.mean(mask_change):.3f}")
        print(f"Assist features detected: {assist_features}")

        return {
            'initial': {'aerial': aerial_initial, 'printed': printed_initial, 'loss': loss_initial},
            'optimized': {'aerial': aerial_optimized, 'printed': printed_optimized, 'loss': loss_optimized},
            'mask_change': mask_change,
            'assist_features': assist_features
        }


def inverse_lithography_optimization(initial_mask, target_image,
                                     learning_rate=ILT_LEARNING_RATE,
                                     max_iterations=ILT_MAX_ITERATIONS):
    """
    逆光刻优化主函数 - 使用解析梯度
    """
    optimizer = InverseLithographyOptimizer()

    # 执行优化
    optimized_mask, history = optimizer.optimize(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=learning_rate,
        max_iterations=max_iterations
    )


    return optimized_mask, history