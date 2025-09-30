import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 导入参数和DMD调制函数
from config.parameters import *
from core.lithography_simulation import dmd_modulation
from utils.image_processing import load_image, binarize_image, save_image



def save_dmd_modulation_results(original_mask, modulated_mask,
                                output_dir="../The_cross-transfer_coefficient/data/output/dmd_test"):
    """
    保存DMD调制结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存原始掩模
    original_img = Image.fromarray((original_mask * 255).astype(np.uint8))
    original_img.save(os.path.join(output_dir, "original_mask.png"))

    # 保存调制后的掩模
    # 归一化调制后的掩模以便显示

    modulated_display = modulated_mask

    modulated_img = Image.fromarray((modulated_display * 255).astype(np.uint8))
    modulated_img.save(os.path.join(output_dir, "modulated_mask.png"))

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_mask, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[0].axis('off')

    axes[1].imshow(modulated_display, cmap='gray')
    axes[1].set_title('DMD Modulated Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dmd_modulation_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Results saved to: {output_dir}")


def test_dmd_modulation():
    """
    测试DMD调制功能
    """
    print("Starting DMD Modulation Test...")

    # 1. 创建测试掩模
    print("Creating test mask...")
    test_mask =load_image(INITIAL_MASK_PATH)
    print(f"Test mask shape: {test_mask.shape}")
    print(f"Test mask value range: [{test_mask.min():.3f}, {test_mask.max():.3f}]")

    # 2. 应用DMD调制
    print("Applying DMD modulation...")
    try:
        modulated_mask = dmd_modulation(
            test_mask,
            Lx=LX, Ly=LY,
            Wx=WX, Wy=WY,
            Tx=TX, Ty=TY,
            dx=DX, dy=DY
        )


        # 3. 保存结果
        print("Saving results...")
        save_dmd_modulation_results(test_mask, modulated_mask)

        print("DMD Modulation Test Completed Successfully!")

    except Exception as e:
        print(f"Error during DMD modulation: {e}")
        import traceback
        traceback.print_exc()


def test_with_real_image():
    """
    使用真实图像测试DMD调制
    """
    from utils.image_processing import load_image

    print("\nTesting with real image...")

    # 加载真实图像
    try:
        real_mask = load_image(INITIAL_MASK_PATH)
        print(f"Real mask shape: {real_mask.shape}")
        print(f"Real mask value range: [{real_mask.min():.3f}, {real_mask.max():.3f}]")

        # 应用DMD调制
        modulated_real = dmd_modulation(
            real_mask,
            Wx=WX, Wy=WY,
            Tx=TX, Ty=TY,
            dx=DX, dy=DY,
            Lx=LX, Ly=LY
        )

        print(f"Modulated real mask shape: {modulated_real.shape}")
        print(f"Modulated real mask value range: [{modulated_real.min():.3f}, {modulated_real.max():.3f}]")

        # 保存结果
        save_dmd_modulation_results(
            real_mask,
            modulated_real,
            output_dir="../The_cross-transfer_coefficient/data/output/dmd_test_real"
        )

        print("Real Image DMD Modulation Test Completed Successfully!")

    except Exception as e:
        print(f"Error with real image test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行测试
    test_dmd_modulation()

    # 可选：使用真实图像测试
    #test_with_real_image()