import matplotlib.pyplot as plt
from config.parameters import *


def plot_comparison(target_image, simulated_image_initial, initial_binary_simulated_image,
                    best_mask, best_simulated_image, optimized_binary_simulated_image,
                    pe_initial, pe_best, save_path=None):
    plt.figure(figsize=(24, 18))

    # 目标图像
    plt.subplot(231)
    plt.imshow(target_image, cmap='gray')
    plt.title('Original Image')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 原始掩膜曝光后的图像
    plt.subplot(232)
    plt.imshow(simulated_image_initial, cmap='gray')
    plt.title('Image after Exposure from Original Mask')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 原始掩膜曝光后的二值图像
    plt.subplot(233)
    plt.imshow(initial_binary_simulated_image, cmap='gray')
    plt.title('Binary Image after Exposure from Original Mask')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    #添加PE文本
    ax = plt.gca()
    text_str=f'PE= {pe_initial:.2f}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes)

    # 优化后的掩膜
    plt.subplot(234)
    plt.imshow(best_mask, cmap='gray')
    plt.title('Optimized Mask')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 优化掩膜曝光后的图像
    plt.subplot(235)
    plt.imshow(best_simulated_image, cmap='gray')
    plt.title('Image after Exposure from Optimized Mask')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 优化掩膜曝光后的二值图像
    plt.subplot(236)
    plt.imshow(optimized_binary_simulated_image, cmap='gray')
    plt.title('Binary Image after Exposure from Optimized Mask')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 添加PE文本
    ax = plt.gca()
    text_str=f'PE= {pe_best:.2f}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_fitness_evolution(log, save_path=None):
    plt.figure()
    minFitnessValues, meanFitnessValues = log.select("min", "avg")
    plt.plot(minFitnessValues, color='red', label='Min Fitness')
    plt.plot(meanFitnessValues, color='green', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average Fitness over Generations')
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()