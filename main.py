import time
import numpy as np
from config.parameters import *
from utils.image_processing import load_image, save_image
from core.lithography_simulation import hopkins_digital_lithography_simulation,photoresist_model
from core.genetic_algorithm import setup_toolbox, run_genetic_algorithm
from utils.visualization import plot_comparison, plot_fitness_evolution


def main():
    # 开始计时
    start_time = time.time()

    # 加载图像float64
    print("Loading images...")
    initial_mask = load_image(INITIAL_MASK_PATH)
    target_image = load_image(TARGET_IMAGE_PATH)

    # 初始掩膜的光刻仿真
    print("Running initial lithography simulation...")
    aerial_image_initial = hopkins_digital_lithography_simulation(initial_mask)
    print_image_initial= photoresist_model(aerial_image_initial)
    PE_initial = np.sum(target_image-print_image_initial)

    # 设置遗传算法工具箱
    print("Setting up genetic algorithm...")
    threshold = 0.5 * np.max(aerial_image_initial)
    toolbox = setup_toolbox(initial_mask.flatten(), target_image, threshold)

    # 运行遗传算法
    print("Running genetic algorithm...")
    pop, hof, log = run_genetic_algorithm(toolbox)

    # 获取最佳个体
    best_mask = np.array(hof[0], dtype=np.float32).reshape((LX, LY))

    # 最佳掩膜的光刻仿真
    print("Running lithography simulation for optimized mask...")
    best_aerial_image = hopkins_digital_lithography_simulation(best_mask)
    best_print_image = photoresist_model(best_aerial_image)
    PE_best = np.sum(target_image-best_print_image)


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

    # 绘制适应度进化图
    plot_fitness_evolution(log, FITNESS_PLOT_PATH)


if __name__ == "__main__":
    main()

