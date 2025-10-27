import matplotlib.pyplot as plt
from config.parameters import *
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

def plot_comparison(initial_mask, simulated_image_initial, initial_binary_simulated_image,
                    best_mask, best_simulated_image, optimized_binary_simulated_image,
                    pe_initial, pe_best, save_path=None):
    plt.figure(figsize=(24, 18))

    # 目标图像
    plt.subplot(231)
    plt.imshow(initial_mask, cmap='gray')
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
    plt.title('Printed image after photoresist from Original Mask')
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
    plt.title('Printed image after photoresist from Optimized Mask')
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


def plot_optimization_history(history, save_path=None):
    """绘制优化历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 损失曲线
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_title('Loss Evolution')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)

    # 掩模演变（显示几个关键迭代）
    if len(history['mask']) > 0:
        num_masks = min(4, len(history['mask']))
        for i in range(num_masks):
            idx = i * len(history['mask']) // num_masks
            axes[0, 1].imshow(history['mask'][idx], cmap='gray',
                              extent=[0, 1, 0, 1], alpha=0.7)
        axes[0, 1].set_title('Mask Evolution')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')

    # 打印图像演变
    if len(history['print_image']) > 0:
        num_images = min(4, len(history['print_image']))
        for i in range(num_images):
            idx = i * len(history['print_image']) // num_images
            axes[1, 0].imshow(history['print_image'][idx], cmap='gray',
                              extent=[0, 1, 0, 1], alpha=0.7)
        axes[1, 0].set_title('Print Image Evolution')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')

    # 最终掩模分布
    if len(history['mask']) > 0:
        axes[1, 1].hist(history['mask'][-1].flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title('Final Mask Distribution')
        axes[1, 1].set_xlabel('Mask Value')
        axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()