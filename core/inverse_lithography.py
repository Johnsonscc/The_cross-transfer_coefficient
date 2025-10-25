# core/inverse_lithography_gradient.py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class GradientBasedILT:
    def __init__(self, learning_rate=0.1, max_iter=100, resist_a=10, resist_Tr=0.225):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.resist_a = resist_a
        self.resist_Tr = resist_Tr

    def photoresist_model(self, intensity):
        """可微分的sigmoid光刻胶模型"""
        return torch.sigmoid(self.resist_a * (intensity - self.resist_Tr))

    def compute_loss(self, print_image, target_image, mask=None, weight_regularization=0.01):
        """计算总损失函数"""
        # L2损失
        l2_loss = F.mse_loss(print_image, target_image)

        # 曲率正则化（可选）
        curvature_loss = 0
        if mask is not None:
            kernel_curv = torch.tensor([[-1 / 16, 5 / 16, -1 / 16],
                                        [5 / 16, -1.0, 5 / 16],
                                        [-1 / 16, 5 / 16, -1 / 16]], dtype=torch.float32)
            curvature = F.conv2d(mask.unsqueeze(0).unsqueeze(0),
                                 kernel_curv.unsqueeze(0).unsqueeze(0), padding=1)
            curvature_loss = torch.mean(curvature ** 2)

        total_loss = l2_loss + weight_regularization * curvature_loss
        return total_loss, l2_loss, curvature_loss

    def optimize(self, initial_mask, target_image, simulation_function):
        """
        梯度下降优化主函数

        Args:
            initial_mask: 初始掩模
            target_image: 目标图像
            simulation_function: 光刻仿真函数
        """
        # 转换为PyTorch张量
        if isinstance(initial_mask, np.ndarray):
            mask = torch.tensor(initial_mask, dtype=torch.float32, requires_grad=True)
        else:
            mask = initial_mask.clone().detach().requires_grad_(True)

        if isinstance(target_image, np.ndarray):
            target = torch.tensor(target_image, dtype=torch.float32)
        else:
            target = target_image.clone().detach()

        # 优化器
        optimizer = torch.optim.Adam([mask], lr=self.learning_rate)
        # 或者使用SGD: optimizer = torch.optim.SGD([mask], lr=self.learning_rate)

        history = {
            'loss': [],
            'l2_loss': [],
            'curvature_loss': [],
            'masks': []
        }

        print("Starting gradient-based ILT optimization...")
        for iteration in tqdm(range(self.max_iter)):
            optimizer.zero_grad()

            # 前向传播：光刻仿真
            aerial_image = simulation_function(mask.detach().numpy())  # 使用你的仿真函数
            aerial_tensor = torch.tensor(aerial_image, dtype=torch.float32)

            # 光刻胶模型
            print_image = self.photoresist_model(aerial_tensor)

            # 计算损失
            total_loss, l2_loss, curvature_loss = self.compute_loss(
                print_image, target, mask
            )

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 投影到[0,1]范围
            with torch.no_grad():
                mask.data = torch.clamp(mask.data, 0, 1)

            # 记录历史
            history['loss'].append(total_loss.item())
            history['l2_loss'].append(l2_loss.item())
            history['curvature_loss'].append(curvature_loss.item())

            if iteration % 10 == 0:
                history['masks'].append(mask.detach().numpy().copy())

            # 提前停止检查
            if iteration > 10 and abs(history['loss'][-1] - history['loss'][-2]) < 1e-6:
                print(f"Converged at iteration {iteration}")
                break

        best_mask = mask.detach().numpy()
        return best_mask, history