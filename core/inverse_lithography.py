import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_gradient(mask, target, singular_values, H_functions, lambda_reg=0.01):
    """
    计算离散梯度表达式 - 完全重写版本
    """
    Ly, Lx = mask.shape

    # 前向传播：计算光强分布
    M_fft = fftshift(fft2(mask))
    intensity = np.zeros((Ly, Lx), dtype=np.float64)
    eigen_images = []

    logger.debug("开始前向传播计算光强")

    for i, (s_val, H_i) in enumerate(zip(singular_values, H_functions)):
        # 频域滤波
        filtered_fft = M_fft * H_i

        # 空间域特征图像
        eigen_image = ifft2(ifftshift(filtered_fft))
        eigen_images.append(eigen_image)

        # 累加光强贡献
        intensity += s_val * np.abs(eigen_image) ** 2

    # 计算损失函数
    data_loss = np.mean((intensity - target) ** 2)

    # 全变分正则化
    tv_loss = compute_total_variation(mask)
    total_loss = data_loss + lambda_reg * tv_loss

    # 反向传播：完全重写梯度计算
    logger.debug("开始反向传播计算梯度")

    # 数据损失梯度部分
    dL_dI = 2 * (intensity - target) / (Lx * Ly)

    # 重新初始化梯度
    gradient = np.zeros((Ly, Lx), dtype=np.float64)

    for i, (s_val, H_i, eigen_img) in enumerate(zip(singular_values, H_functions, eigen_images)):
        # 正确计算伴随场：dL/dE_i = dL/dI * s_val * 2 * E_i^*
        dL_dE_i = dL_dI * s_val * 2 * eigen_img.conj()

        # 转换到频域
        dL_dE_i_fft = fftshift(fft2(dL_dE_i))

        # 应用伴随光学传递函数：dL/dM_fft = H_i^* * dL/dE_i_fft
        dL_dM_fft_i = np.conj(H_i) * dL_dE_i_fft

        # 转换回空间域并累加实部
        dL_dM_i = ifft2(ifftshift(dL_dM_fft_i))
        gradient += np.real(dL_dM_i)

    # 添加正则化梯度
    reg_gradient = compute_tv_gradient(mask)
    gradient += lambda_reg * reg_gradient

    # 梯度检查
    grad_norm = np.linalg.norm(gradient)
    logger.debug(f"梯度计算完成: 范数={grad_norm:.6f}")

    return gradient, total_loss


def compute_total_variation(mask):
    """计算全变分正则化损失"""
    dx = np.diff(mask, axis=1)
    dy = np.diff(mask, axis=0)
    tv_loss = np.sum(dx ** 2) + np.sum(dy ** 2)
    return tv_loss / mask.size


def compute_tv_gradient(mask):
    """计算全变分正则化的梯度"""
    Ly, Lx = mask.shape
    gradient = np.zeros((Ly, Lx))

    # 简化但有效的TV梯度计算
    # x方向
    grad_x = np.zeros_like(mask)
    grad_x[:, 1:-1] = mask[:, 2:] - 2 * mask[:, 1:-1] + mask[:, :-2]
    grad_x[:, 0] = mask[:, 1] - mask[:, 0]
    grad_x[:, -1] = mask[:, -1] - mask[:, -2]

    # y方向
    grad_y = np.zeros_like(mask)
    grad_y[1:-1, :] = mask[2:, :] - 2 * mask[1:-1, :] + mask[:-2, :]
    grad_y[0, :] = mask[1, :] - mask[0, :]
    grad_y[-1, :] = mask[-1, :] - mask[-2, :]

    gradient = 2 * (grad_x + grad_y) / (Lx * Ly)
    return gradient


def gradient_descent_optimization(initial_mask, target, singular_values, H_functions,
                                  learning_rate=0.1, iterations=100, lambda_reg=0.01,
                                  patience=20, min_delta=1e-6):
    """
    基于梯度下降的逆光刻优化 - 完全重写版本
    """
    logger.info("开始梯度下降优化")

    mask = initial_mask.copy().astype(np.float64)
    loss_history = []
    best_loss = float('inf')
    best_mask = mask.copy()
    no_improvement_count = 0

    # 进度条
    pbar = tqdm(range(iterations), desc="ILT优化进度")

    for iteration in pbar:
        try:
            # 计算梯度和损失
            gradient, current_loss = compute_gradient(mask, target, singular_values,
                                                      H_functions, lambda_reg)

            # 检查梯度是否为0
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < 1e-12:
                logger.warning(f"梯度接近0 ({grad_norm:.2e})，添加随机扰动")
                gradient += 1e-6 * np.random.randn(*gradient.shape)
                grad_norm = np.linalg.norm(gradient)

            # 梯度下降更新 - 添加动量项
            mask_update = learning_rate * gradient / (grad_norm + 1e-8)  # 归一化
            mask -= mask_update

            # 投影到可行集 [0, 1]
            mask = np.clip(mask, 0, 1)

            # 记录损失
            loss_history.append(current_loss)

            # 早停检查
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                best_mask = mask.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # 更新进度条
            pbar.set_postfix({
                '损失': f'{current_loss:.6f}',
                '最佳损失': f'{best_loss:.6f}',
                '梯度': f'{grad_norm:.2e}'
            })

            # 早停条件
            if no_improvement_count >= patience and iteration > 10:
                logger.info(f"早停在迭代 {iteration}, 最佳损失: {best_loss:.6f}")
                break

        except Exception as e:
            logger.error(f"迭代 {iteration} 出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            break

    logger.info(f"优化完成，最终损失: {best_loss:.6f}")
    return best_mask, loss_history


def inverse_lithography_optimization(initial_mask, target_image,
                                     lambda_=405, na=0.5, sigma=0.5, dx=7560, dy=7560,
                                     learning_rate=0.1, iterations=100, k_svd=5, **kwargs):
    """
    逆光刻优化主函数 - 修复版本
    """
    logger.info("初始化逆光刻优化")

    # 验证输入
    assert initial_mask.shape == target_image.shape, "掩模和目标图像尺寸不匹配"

    # 导入必要的函数
    from core.lithography_simulation import compute_tcc_svd, light_source_function, pupil_response_function

    # 计算频域坐标和TCC SVD
    Ly, Lx = initial_mask.shape
    fx = np.linspace(-0.5 / dx, 0.5 / dx, Lx)
    fy = np.linspace(-0.5 / dy, 0.5 / dy, Ly)

    J = lambda fx, fy: light_source_function(fx, fy, sigma, na, lambda_)
    P = lambda fx, fy: pupil_response_function(fx, fy, na, lambda_)

    logger.info("计算TCC SVD分解")
    singular_values, H_functions = compute_tcc_svd(J, P, fx, fy, k_svd)

    # 运行优化
    optimized_mask, loss_history = gradient_descent_optimization(
        initial_mask, target_image, singular_values, H_functions,
        learning_rate=learning_rate, iterations=iterations, **kwargs
    )

    return optimized_mask, loss_history


