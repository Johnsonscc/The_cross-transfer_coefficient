import time
import numpy as np
from config.parameters import *
from utils.image_processing import load_image, save_image
from core.lithography_simulation import hopkins_digital_lithography_simulation, photoresist_model
from core.inverse_lithography import inverse_lithography_optimization,InverseLithographyOptimizer
from utils.visualization import plot_comparison,plot_optimization_history


def main():
    # 开始计时
    start_time = time.time()

    # 加载图像
    print("Loading images...")
    initial_mask = load_image(INITIAL_MASK_PATH)
    target_image = load_image(TARGET_IMAGE_PATH)

    # 初始掩膜的光刻仿真
    print("Running initial lithography simulation...")
    aerial_image_initial = hopkins_digital_lithography_simulation(initial_mask)
    print_image_initial = photoresist_model(aerial_image_initial)
    PE_initial = np.sum((target_image - print_image_initial) ** 2)

    # 创建优化器并检查梯度
    optimizer = InverseLithographyOptimizer(
        lambda_=LAMBDA, na=NA, sigma=SIGMA, dx=DX, dy=DY,
        lx=LX, ly=LY, a=A, Tr=TR, k_svd=10
    )
    # 梯度检查
    #analytical_grad, numerical_grad = optimizer.gradient_check(initial_mask, target_image)

    # 使用逆光刻优化
    print("Starting inverse lithography optimization...")
    best_mask, optimization_history = inverse_lithography_optimization(
        initial_mask=initial_mask,
        target_image=target_image,
        learning_rate=0.05,
        iterations=30,  # 少量迭代测试
        gradient_method="smart",  # 使用智能采样
        sample_ratio=0.2,  # 采样20%的像素
        lambda_=LAMBDA,
        na=NA,
        sigma=SIGMA,
        dx=DX,
        dy=DY,
        lx=target_image.shape[0],
        ly=target_image.shape[1],
        a=A,
        Tr=TR,
        k_svd=5  # 减少SVD模式以加速
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

    # 保存优化后的掩膜
    save_image(best_mask, OUTPUT_MASK_PATH)

    # 可视化结果
    plot_comparison(target_image, aerial_image_initial, print_image_initial,
                    best_mask, best_aerial_image, best_print_image,
                    PE_initial, PE_best, RESULTS_IMAGE_PATH)

    # 绘制优化历史
    plot_optimization_history(optimization_history, FITNESS_PLOT_PATH)


if __name__ == "__main__":
    main()