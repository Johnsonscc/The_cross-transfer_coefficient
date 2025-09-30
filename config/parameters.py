# 参数配置
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

# 光刻仿真参数
LAMBDA = 405  # 波长（单位：纳米）
Z = 803000000  # 距离（单位：纳米）
DX = DY = 7560  # 像素尺寸（单位：纳米）
LX = LY = 10000  # 图像尺寸（单位：像素）
N = 1.5  # 折射率（无量纲）
SIGMA = 0.5  # 部分相干因子（无量纲）
NA = 0.5  # 数值孔径（无量纲）

# DMD调制参数
WX = 7560  # 微镜宽度（单位：纳米）
WY = 7560  # 微镜高度（单位：纳米）
TX = 8560  # 微镜周期（x方向）（单位：纳米）
TY = 8560  # 微镜周期（y方向）（单位：纳米）


# 遗传算法参数
POPULATION_SIZE = 200
GENERATIONS = 100
CXPB = 0.4  # 交叉概率
MUTPB = 0.4  # 变异概率
NOISE_SCALE = 0.02  # 初始噪声比例
TOURNSIZE = 3  # 锦标赛选择的大小
INDPB = 0.02  # 个体变异概率

# 文件路径
INITIAL_MASK_PATH = "../The_cross-transfer_coefficient/data/input/mask04.png"
TARGET_IMAGE_PATH = "../The_cross-transfer_coefficient/data/input/mask04.png"
OUTPUT_MASK_PATH = "../The_cross-transfer_coefficient/data/output/optimized_mask_hexagonal.png"
RESULTS_IMAGE_PATH = "../The_cross-transfer_coefficient/data/output/results_comparison_hexagonal.png"
FITNESS_PLOT_PATH = "../The_cross-transfer_coefficient/data/output/fitness_evolution_hexagonal.png"
SAVE_PATH="../The_cross-transfer_coefficient/data/output/save.png"

# 可视化参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'