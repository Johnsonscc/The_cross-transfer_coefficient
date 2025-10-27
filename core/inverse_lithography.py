# [file name]: inverse_lithography.py
import numpy as np
from tqdm import tqdm
from core.lithography_simulation import hopkins_digital_lithography_simulation, photoresist_model


class AdamOptimizer:
    """Adam优化器实现"""

    def __init__(self, params_shape, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        # 初始化一阶和二阶矩估计
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)

    def step(self, gradient):
        """执行一步Adam更新"""
        self.t += 1

        # 更新一阶矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

        # 更新二阶矩估计
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 参数更新
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return update


class InverseLithographyOptimizer:
    def __init__(self, lambda_=405, na=0.5, sigma=0.5, dx=7560, dy=7560,
                 lx=30, ly=30, a=10, Tr=0.5, k_svd=10):
        self.lambda_ = lambda_
        self.na = na
        self.sigma = sigma
        self.dx = dx
        self.dy = dy
        self.lx = lx
        self.ly = ly
        self.a = a
        self.Tr = Tr
        self.k_svd = k_svd

    def compute_loss(self, mask, target):
        """计算损失函数 - 直接调用原始仿真函数"""
        # 使用原始的Hopkins仿真和光刻胶模型
        aerial_image = hopkins_digital_lithography_simulation(
            mask,
            lambda_=self.lambda_,
            lx=self.lx,
            ly=self.ly,
            dx=self.dx,
            dy=self.dy,
            sigma=self.sigma,
            na=self.na,
            k_svd=self.k_svd
        )
        print_image = photoresist_model(aerial_image, a=self.a, Tr=self.Tr)

        # 均方误差损失
        loss = np.mean((print_image - target) ** 2)

        return loss, aerial_image, print_image

    def compute_numerical_gradient_batch(self, mask, target, batch_size=100, epsilon=1e-6):
        """批量计算数值梯度 - 减少仿真调用次数"""
        print("Computing numerical gradient...")
        grad = np.zeros_like(mask)
        original_loss, _, _ = self.compute_loss(mask, target)

        # 创建所有需要扰动的位置
        positions = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                positions.append((i, j))

        # 分批处理
        total_positions = len(positions)
        num_batches = (total_positions + batch_size - 1) // batch_size

        print(f"Total positions: {total_positions}, Batch size: {batch_size}, Batches: {num_batches}")

        for batch_idx in tqdm(range(num_batches), desc="Gradient Batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_positions)
            batch_positions = positions[start_idx:end_idx]

            # 为当前批次创建扰动掩模
            mask_plus_batch = np.tile(mask, (len(batch_positions), 1, 1))

            for k, (i, j) in enumerate(batch_positions):
                mask_plus_batch[k, i, j] += epsilon

            # 批量计算损失
            batch_losses = []
            for k in range(len(batch_positions)):
                loss_plus, _, _ = self.compute_loss(mask_plus_batch[k], target)
                batch_losses.append(loss_plus)

            # 计算梯度
            for k, (i, j) in enumerate(batch_positions):
                grad[i, j] = (batch_losses[k] - original_loss) / epsilon

        return grad

    def compute_numerical_gradient_smart(self, mask, target, sample_ratio=0.3, epsilon=1e-6):
        """智能数值梯度计算 - 只计算部分像素的梯度"""
        print("Computing smart numerical gradient...")
        grad = np.zeros_like(mask)
        original_loss, _, _ = self.compute_loss(mask, target)

        # 计算需要采样的像素数量
        total_pixels = mask.shape[0] * mask.shape[1]
        sample_count = int(total_pixels * sample_ratio)

        # 随机选择像素位置
        positions = []
        all_positions = [(i, j) for i in range(mask.shape[0]) for j in range(mask.shape[1])]
        indices = np.random.choice(len(all_positions), size=sample_count, replace=False)

        for idx in indices:
            positions.append(all_positions[idx])

        print(f"Sampling {len(positions)} out of {total_pixels} pixels ({sample_ratio * 100:.1f}%)")

        # 计算选中位置的梯度
        for i, j in tqdm(positions, desc="Gradient Computation"):
            # 正向扰动
            mask_plus = mask.copy()
            mask_plus[i, j] += epsilon
            loss_plus, _, _ = self.compute_loss(mask_plus, target)

            # 数值梯度
            grad[i, j] = (loss_plus - original_loss) / epsilon

        # 对未采样的像素使用插值
        if sample_ratio < 1.0:
            from scipy import ndimage
            # 创建采样掩码
            sampled_mask = np.zeros_like(mask, dtype=bool)
            for i, j in positions:
                sampled_mask[i, j] = True

            # 使用高斯滤波进行插值
            grad_smoothed = ndimage.gaussian_filter(grad, sigma=1.0)

            # 将采样点的梯度替换回精确值
            grad = np.where(sampled_mask, grad, grad_smoothed)

        return grad

    def optimize_with_adam(self, initial_mask, target,
                           learning_rate=0.01, iterations=50,  # 减少迭代次数
                           beta1=0.9, beta2=0.999, epsilon=1e-8,
                           regularization_weight=0.001,
                           gradient_method="smart",  # "smart" or "batch"
                           sample_ratio=0.3,
                           batch_size=50,
                           progress_bar=True):
        """使用Adam优化器和数值梯度进行优化"""
        print(f"Starting ILT optimization with {gradient_method} gradient method...")

        mask = initial_mask.copy()
        adam = AdamOptimizer(mask.shape, lr=learning_rate,
                             beta1=beta1, beta2=beta2, epsilon=epsilon)

        history = {
            'loss': [],
            'mask_range': [],
            'aerial_images': [],
            'print_images': [],
            'masks': []
        }

        best_loss = float('inf')
        best_mask = mask.copy()
        best_aerial = None
        best_print = None

        if progress_bar:
            pbar = tqdm(range(iterations), desc="ILT Optimization")
        else:
            pbar = range(iterations)

        for iteration in pbar:
            try:
                # 计算当前损失
                current_loss, aerial_image, print_image = self.compute_loss(mask, target)

                # 选择梯度计算方法
                if gradient_method == "batch":
                    gradient = self.compute_numerical_gradient_batch(mask, target,
                                                                     batch_size=batch_size)
                else:  # "smart"
                    gradient = self.compute_numerical_gradient_smart(mask, target,
                                                                     sample_ratio=sample_ratio)

                # 添加正则化
                reg_gradient = 2 * regularization_weight * (mask - 0.5)
                total_gradient = gradient + reg_gradient

                # Adam更新
                update = adam.step(total_gradient)
                mask = mask - update

                # 投影到可行集
                mask = np.clip(mask, 0, 1)

                # 更新最佳结果
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_mask = mask.copy()
                    best_aerial = aerial_image.copy()
                    best_print = print_image.copy()

                # 记录历史
                history['loss'].append(current_loss)
                history['mask_range'].append((np.min(mask), np.max(mask)))

                # 每5次迭代保存完整状态
                if iteration % 5 == 0:
                    history['aerial_images'].append(aerial_image.copy())
                    history['print_images'].append(print_image.copy())
                    history['masks'].append(mask.copy())

                # 更新进度条
                if progress_bar:
                    pbar.set_postfix({
                        'Loss': f'{current_loss:.6f}',
                        'Best': f'{best_loss:.6f}'
                    })

                # 定期打印进展
                if iteration % 10 == 0:
                    improvement = history['loss'][0] - current_loss if history['loss'] else 0
                    print(f"Iteration {iteration}: Loss = {current_loss:.6f}, Improvement = {improvement:.6f}")

            except Exception as e:
                print(f"Error at iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                break

        # 添加最佳结果到历史
        history['best_mask'] = best_mask
        history['best_aerial'] = best_aerial
        history['best_print'] = best_print

        print(f"Optimization completed. Best loss: {best_loss:.6f}")
        return best_mask, history


def inverse_lithography_optimization(initial_mask, target_image,
                                     learning_rate=0.01, iterations=50,
                                     gradient_method="smart",
                                     sample_ratio=0.3,
                                     batch_size=50,
                                     **kwargs):
    """
    逆光刻优化主函数 - 直接使用原始仿真函数
    """
    optimizer = InverseLithographyOptimizer(**kwargs)

    best_mask, history = optimizer.optimize_with_adam(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=learning_rate,
        iterations=iterations,
        gradient_method=gradient_method,
        sample_ratio=sample_ratio,
        batch_size=batch_size
    )

    return best_mask, history