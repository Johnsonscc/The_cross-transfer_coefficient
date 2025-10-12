import time
import numpy as np
from config.parameters import *
from utils.image_processing import load_image, binarize_image, save_image
from core.lithography_simulation_gpu import hopkins_digital_lithography_simulation
from core.genetic_algorithm import setup_toolbox, run_genetic_algorithm
from utils.visualization import plot_comparison, plot_fitness_evolution


def main():
    # 开始计时
    start_time = time.time()

    # 加载图像
    print("Loading images...")
    initial_mask = load_image(INITIAL_MASK_PATH)
    target_image = load_image(TARGET_IMAGE_PATH)

    # 初始掩膜的光刻仿真
    print("Running initial lithography simulation...")
    simulated_image_initial = hopkins_digital_lithography_simulation(initial_mask)
    threshold = 0.2 * np.max(simulated_image_initial)
    binary_image_initial = binarize_image(simulated_image_initial, threshold)
    PE_initial = np.sum(np.abs(binary_image_initial.astype(np.float32) - target_image.astype(np.float32)))

    # 设置遗传算法工具箱
    print("Setting up genetic algorithm...")
    toolbox = setup_toolbox(initial_mask.flatten(), target_image, threshold)

    # 运行遗传算法
    print("Running genetic algorithm...")
    pop, hof, log = run_genetic_algorithm(toolbox)

    # 获取最佳个体
    best_mask = np.array(hof[0], dtype=np.float32).reshape((LX, LY))

    # 最佳掩膜的光刻仿真
    print("Running lithography simulation for optimized mask...")
    best_simulated_image = best_mask#           暂时注释遗传算法的掩模优化
    best_binary_image = binarize_image(best_simulated_image, threshold)
    PE_best = np.sum(np.abs(best_binary_image.astype(np.float32) - target_image.astype(np.float32)))


    # 结束计时
    end_time = time.time()
    print(f'Running time: {end_time - start_time:.3f} seconds')
    print(f'Initial PE: {PE_initial}, Best PE: {PE_best}')

    # 保存优化后的掩膜
    save_image(best_mask, OUTPUT_MASK_PATH)

    # 准备可视化数据
    initial_binary_simulated_image = binarize_image(simulated_image_initial, threshold)
    optimized_binary_simulated_image = binarize_image(best_simulated_image, threshold)

    # 可视化结果
    plot_comparison(target_image, simulated_image_initial, initial_binary_simulated_image,
                    best_mask, best_simulated_image, optimized_binary_simulated_image,
                    PE_initial, PE_best, RESULTS_IMAGE_PATH)

    # 绘制适应度进化图
    plot_fitness_evolution(log, FITNESS_PLOT_PATH)


if __name__ == "__main__":
    main()

