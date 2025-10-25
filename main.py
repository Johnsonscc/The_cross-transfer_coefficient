# main.py (修改版本)
import time
import numpy as np
import torch
from config.parameters import *
from utils.image_processing import load_image, save_image
from core.lithography_simulation import hopkins_digital_lithography_simulation, photoresist_model
from core.inverse_lithography import GradientBasedILT  # 新的梯度优化器
from utils.visualization import plot_comparison


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 开始计时
    start_time = time.time()

    # 加载图像
    print("Loading images...")
    initial_mask = load_image(INITIAL_MASK_PATH)
    target_image = load_image(TARGET_IMAGE_PATH)

    # 包装仿真函数以兼容PyTorch
    def simulation_wrapper(mask_array):
        """包装仿真函数，处理numpy数组输入输出"""
        aerial_image = hopkins_digital_lithography_simulation(mask_array)
        return aerial_image

    # 初始掩膜的光刻仿真
    print("Running initial lithography simulation...")
    aerial_image_initial = hopkins_digital_lithography_simulation(initial_mask)
    print_image_initial = photoresist_model(aerial_image_initial)
    PE_initial = np.sum((target_image - print_image_initial) ** 2)

    # 使用梯度下降逆光刻优化
    print("Starting gradient-based inverse lithography optimization...")

    # 创建优化器实例
    ilt_optimizer = GradientBasedILT(
        learning_rate=ILT_LEARNING_RATE,
        max_iter=ILT_MAX_ITER,
        resist_a=ILT_RESIST_A,
        resist_Tr=ILT_RESIST_Tr
    )

    # 执行优化
    best_mask, optimization_history = ilt_optimizer.optimize(
        initial_mask=initial_mask,
        target_image=target_image,
        simulation_function=simulation_wrapper
    )

    # 最佳掩膜的光刻仿真
    print("Running lithography simulation for optimized mask...")
    best_aerial_image = hopkins_digital_lithography_simulation(best_mask)
    best_print_image = photoresist_model(best_aerial_image)
    PE_best = np.sum((target_image - best_print_image) ** 2)

    # 结束计时
    end_time = time.time()
    print(f'Running time: {end_time - start_time:.3f} seconds')
    print(f'Initial PE: {PE_initial}, Best PE: {PE_best}')
    print(f'Improvement: {(PE_initial - PE_best) / PE_initial * 100:.2f}%')

    # 保存优化后的掩膜
    save_image(best_mask, OUTPUT_MASK_PATH)

    # 保存优化历史
    np.savez('optimization_history.npz', **optimization_history)

    # 可视化结果
    plot_comparison(target_image, aerial_image_initial, print_image_initial,
                    best_mask, best_aerial_image, best_print_image,
                    PE_initial, PE_best, RESULTS_IMAGE_PATH)

    # 绘制损失曲线
    plot_loss_curve(optimization_history)


def plot_loss_curve(history):
    """绘制损失函数下降曲线"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Total Loss', linewidth=2)
    plt.plot(history['l2_loss'], label='L2 Loss', linestyle='--')
    plt.plot(history['curvature_loss'], label='Curvature Loss', linestyle=':')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Gradient-Based ILT Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()