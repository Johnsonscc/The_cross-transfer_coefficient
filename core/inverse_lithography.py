import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
import logging
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from config.parameters import *

logger = logging.getLogger(__name__)


class InverseLithographyOptimizer:
    """
    逆光刻优化器 基于解析梯度计算
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA, dx=DX, dy=DY,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR):
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

    def pupil_response_function(fx, fy, na=NA, lambda_=LAMBDA):
        r = np.sqrt(fx ** 2 + fy ** 2)
        r_max = na / lambda_
        P = np.where(r < r_max, lambda_ ** 2 / (np.pi * (na) ** 2), 0)
        return P

    def _compute_full_tcc_matrix(self, fx, fy, sparsity_threshold=0.001):

        # 创建频域网格
        Lx, Ly = len(fx), len(fy)
        FX, FY = np.meshgrid(fx, fy, indexing='xy')


        # 计算光源函数 J(f,g)
        r_j = np.sqrt(FX ** 2 + FY ** 2)
        r_max_j = self.sigma * self.na / self.lambda_
        J = np.where(r_j <= r_max_j,
                     self.lambda_ ** 2 / (np.pi * (self.sigma * self.na) ** 2), 0.0)

        # 计算瞳函数 P(f,g)
        r_p = np.sqrt(FX ** 2 + FY ** 2)
        r_max_p = self.na / self.lambda_
        P = np.where(r_p <= r_max_p,
                     self.lambda_ ** 2 / (np.pi * (self.na) ** 2), 0.0)

        tcc_kernel = J * P

        print(f"Building 4D TCC matrix ({Lx}x{Ly}x{Lx}x{Ly})...")

        # 在邻域搜索范围计算频率相互作用
        TCC_sparse = lil_matrix((Lx * Ly, Lx * Ly), dtype=np.complex128)
        neighborhood_radius = 10

        for i in tqdm(range(Lx), desc="Improved TCC Construction"):
            for j in range(Ly):
                # 计算核函数大于阈值的频率
                if np.abs(tcc_kernel[i, j]) > sparsity_threshold:
                    for m in range(max(0, i - neighborhood_radius), min(Lx, i + neighborhood_radius + 1)):
                        for n in range(max(0, j - neighborhood_radius), min(Ly, j + neighborhood_radius + 1)):
                            if np.abs(tcc_kernel[m, n]) > sparsity_threshold:
                                idx1 = i * Ly + j
                                idx2 = m * Ly + n
                                TCC_sparse[idx1, idx2] = tcc_kernel[i, j] * np.conj(tcc_kernel[m, n])

        TCC_csr = csr_matrix(TCC_sparse)

        return TCC_csr

    def _svd_of_tcc_matrix(self, TCC_csr, k, Lx=LX, Ly=LY):

        print("Performing SVD decomposition...")
        k_actual = min(k, min(TCC_csr.shape) - 1)

        U, S, Vh = svds(TCC_csr, k=k_actual)

        # 过滤掉太小的奇异值
        significant_mask = S > (np.max(S) * 0.01)  # 只保留大于1%最大值的奇异值
        S = S[significant_mask]
        U = U[:, significant_mask]

        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]

        H_functions = []
        for i in range(len(S)):
            H_i = U[:, i].reshape(Lx, Ly)
            H_functions.append(H_i)

        return S, H_functions

    def _precompute_tcc_svd(self):

        # 频域坐标
        max_freq = self.na /self.lambda_
        freq = 2 * max_freq

        fx = np.linspace(-freq, freq, LX)
        fy = np.linspace(-freq, freq, LY)

        # 完整计算TCC矩阵
        TCC_4d = self._compute_full_tcc_matrix(fx, fy)

        # 对TCC矩阵进行SVD分解
        self.singular_values, self.eigen_functions = self._svd_of_tcc_matrix(TCC_4d, self.k_svd)

        print(f"TCC SVD precomputation completed with {len(self.singular_values)} singular values")

    def photoresist_model(self, intensity):
        #光刻胶模型 - sigmoid函数
        return 1 / (1 + np.exp(-self.a * (intensity - self.tr)))

    def _compute_analytical_gradient(self, mask, target):
        #F(ω) = ∑ [z_ξ - 1/(1+exp(-a(∑σ_i|h_i⊗M|² - T_r)))]²

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

        # 1. ∂F/∂P = -2(z - P)
        dF_dP = -2 * (target - P)

        # 2. ∂P/∂I = a * P * (1 - P) * (∂I_norm/∂I)
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
        # ∂I/∂M = 2 * σ_i * Re{A_i * (∂(h_i⊗M)/∂M)}
        for i, (s_val, H_i, A_i, I_i) in enumerate(zip(
                self.singular_values, self.eigen_functions, A_i_list, I_i_list
        )):
            # 计算 ∂F/∂A_i = ∂F/∂I * ∂I/∂A_i = dF_dP * dP_dI * 2 * σ_i * A_i
            dF_dA_i = dF_dP * dP_dI * 2 * s_val * A_i.conj()

            # 通过傅里叶变换计算梯度贡献
            dF_dA_i_fft = fftshift(fft2(dF_dA_i))
            gradient_contribution = ifft2(ifftshift(dF_dA_i_fft * np.conj(H_i)))

            gradient += gradient_contribution

        # 取实部，因为掩模是实数
        gradient_real = np.real(gradient)

        return loss, gradient_real, intensity_norm, P

    def optimize(self, initial_mask, target, learning_rate, max_iterations):
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


        best_mask = mask.copy()
        best_loss = float('inf')

        for iteration in range(max_iterations):
            # 使用解析梯度计算损失和梯度
            loss, gradient, aerial_image, printed_image = self._compute_analytical_gradient(mask, target)

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


def inverse_lithography_optimization(initial_mask, target_image,
                                     learning_rate=ILT_LEARNING_RATE,
                                     max_iterations=ILT_MAX_ITERATIONS):
    #逆光刻优化主函数 - 使用解析梯度
    optimizer = InverseLithographyOptimizer()

    # 执行优化
    optimized_mask, history = optimizer.optimize(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=learning_rate,
        max_iterations=max_iterations
    )


    return optimized_mask, history