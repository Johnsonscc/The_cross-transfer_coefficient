import time
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# 光刻仿真参数
LAMBDA = 405  # 波长（单位：纳米）
Z = 803000000  # 距离（单位：纳米）774
DX = DY = 7560  # 像素尺寸（单位：纳米）
LX = LY = 1000  # 图像尺寸（单位：像素）
N = 1.5  # 折射率（无量纲）1.4
SIGMA = 0.5  # 部分相干因子（无量纲）
NA = 0.5  # 数值孔径（无量纲）

# 光刻胶参数
A = 10.0            # sigmoid函数梯度
TR = 0.5            # 阈值参数

# 文件路径
INITIAL_MASK_PATH = "../lithography_simulation_Hopkins/data/input/cell1000.png"
TARGET_IMAGE_PATH = "../lithography_simulation_Hopkins/data/input/cell1000.png"
OUTPUT_MASK_PATH = "../lithography_simulation_Hopkins/data/output/optimized_mask_cell1000.png"
RESULTS_IMAGE_PATH = "../lithography_simulation_Hopkins/data/output/results_comparison_cell1000.png"
FITNESS_PLOT_PATH = "../lithography_simulation_Hopkins/data/output/fitness_evolution_cell1000.png"

def load_image(path,grayscale=True):
    image = iio.imread(path)
    if grayscale and len(image.shape)>2:#将彩色通道图像转化为灰度图
        image = rgb2gray(image)
    return image


def save_image(image,path):
    plt.imsave(path, image, cmap='gray', vmin=image.min(), vmax=image.max())#保存灰度图像

def light_source_function(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = sigma * na / lambda_
    J = np.where(r <= r_max, lambda_ ** 2 / (np.pi * (sigma * na) ** 2), 0)
    return J


def pupil_response_function(fx, fy, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = na / lambda_
    P = np.where(r < r_max, lambda_ ** 2 / (np.pi * (na) ** 2), 0)
    return P


def compute_tcc_svd(J, P, fx, fy, k, sparsity_threshold=0.001):
    """
    改进的TCC计算，确保包含足够的高频成分
    """
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    J_vals = J(FX, FY)
    P_vals = P(FX, FY)

    # 确保瞳函数包含适当的高频截止
    tcc_kernel = J_vals * P_vals

    Lx, Ly = len(fx), len(fy)

    print("Building improved TCC matrix...")

    # 使用更密集的采样确保高频信息
    from scipy.sparse import lil_matrix, csr_matrix
    TCC_sparse = lil_matrix((Lx * Ly, Lx * Ly), dtype=np.complex128)

    # 扩大邻域搜索范围以捕捉更多频率相互作用
    neighborhood_radius = 10  # 从5增加到10

    for i in tqdm(range(Lx), desc="Improved TCC Construction"):
        for j in range(Ly):
            if np.abs(tcc_kernel[i, j]) > sparsity_threshold:
                for m in range(max(0, i - neighborhood_radius),
                               min(Lx, i + neighborhood_radius + 1)):
                    for n in range(max(0, j - neighborhood_radius),
                                   min(Ly, j + neighborhood_radius + 1)):
                        if np.abs(tcc_kernel[m, n]) > sparsity_threshold:
                            idx1 = i * Ly + j
                            idx2 = m * Ly + n
                            TCC_sparse[idx1, idx2] = tcc_kernel[i, j] * np.conj(tcc_kernel[m, n])

    TCC_csr = csr_matrix(TCC_sparse)


    print(f"Performing SVD with {k} components...")
    U, S, Vh = svds(TCC_csr, k=k)

    # 过滤掉太小的奇异值
    significant_mask = S > (np.max(S) * 0.01)  # 只保留大于1%最大值的奇异值
    S = S[significant_mask]
    U = U[:, significant_mask]

    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]

    H_functions = []
    for i in tqdm(range(len(S)), desc="Extracting eigenfunctions"):
        H_i = U[:, i].reshape(Lx, Ly)
        H_functions.append(H_i)

    print(f"Using {len(S)} significant singular values")
    return S, H_functions


def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                           dx=DX, dy=DY, sigma=SIGMA, na=NA, k_svd=10):
    # 频域坐标
    fx = np.linspace(-0.5 / dx, 0.5 / dx, lx)
    fy = np.linspace(-0.5 / dy, 0.5 / dy, ly)

    # 定义光源和瞳函数
    J = lambda fx, fy: light_source_function(fx, fy, sigma, na, lambda_)
    P = lambda fx, fy: pupil_response_function(fx, fy, na, lambda_)

    # 计算TCC并进行SVD分解
    singular, H_functions = compute_tcc_svd(J, P, fx, fy, k_svd)

    # 掩模的傅里叶变换
    M_fft = fftshift(fft2(mask))

    # 初始化光强
    intensity = np.zeros((lx, ly), dtype=np.float64)

    # 根据SVD分解计算光强
    print(f"Computing intensity using {len(singular)} singular values...")
    for i, (s_val, H_i) in enumerate(zip(singular, H_functions)):
        # 滤波后的频谱
        filtered_fft = M_fft * H_i

        # 逆傅里叶变换
        filtered_space = ifft2(ifftshift(filtered_fft))

        # 累加光强贡献
        intensity += s_val * np.abs(filtered_space) ** 2

    # 最终结果
    result = intensity

    # 修复归一化：避免除0
    intensity_min = np.min(intensity)
    intensity_max = np.max(intensity)

    if intensity_max - intensity_min > 1e-10:
        result = (intensity - intensity_min) / (intensity_max - intensity_min)
    else:
        print("警告: 光强分布范围过小，使用备选归一化")
        result = intensity / (intensity_max + 1e-10)  # 避免除0

    return result


def photoresist_model(intensity, a=15.0, Tr=0.5, diffusion_length=10.0):

    resist_pattern = 1 / (1 + np.exp(-a * (intensity - Tr)))

    # 添加扩散效应模拟化学放大
    resist_pattern = gaussian_filter(resist_pattern, sigma=diffusion_length / 10.0)

    return resist_pattern

def plot_comparison(target_image, aerial_image_initial, print_image_initial,
                    pe_initial, save_path=None):
    """简化比较图 - 只显示关键结果"""
    plt.figure(figsize=(18, 12))

    # 目标图像
    plt.subplot(231)
    plt.imshow(target_image, cmap='gray')
    plt.title('Target Image')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 原始掩膜曝光后的图像
    plt.subplot(232)
    plt.imshow(aerial_image_initial, cmap='gray')
    plt.title('Aerial Image (Original)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 原始掩膜曝光后的值图像
    plt.subplot(233)
    plt.imshow(print_image_initial, cmap='gray')
    plt.title('Printed Image (Original)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.text(0.05, 0.95, f'PE = {pe_initial:.2f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


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
    PE_aerial=np.sum((target_image-aerial_image_initial)**2)
    PE_initial = np.sum((target_image - print_image_initial) ** 2)

    end_time = time.time()
    print(f'Running time: {end_time - start_time:.3f} seconds')
    print(f'Aerial PE :{PE_aerial}, Initial PE: {PE_initial}')

    # 可视化结果
    plot_comparison(target_image, aerial_image_initial, print_image_initial,PE_initial,RESULTS_IMAGE_PATH)


if __name__ == "__main__":
    main()
