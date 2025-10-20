# 参数配置
import matplotlib.pyplot as plt

# 光刻仿真参数
LAMBDA = 405  # 波长（单位：纳米）
Z = 803000000  # 距离（单位：纳米）
DX = DY = 7560  # 像素尺寸（单位：纳米）
LX = LY = 30  # 图像尺寸（单位：像素）
N = 1.5  # 折射率（无量纲）
SIGMA = 0.5  # 部分相干因子（无量纲）
NA = 0.5  # 数值孔径（无量纲）

# 光刻胶参数
A = 10.0            # sigmoid函数梯度
TR = 0.5            # 阈值参数

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

# 逆光刻优化参数
ILT_LEARNING_RATE = 0.1
ILT_MAX_ITER = 100
ILT_RESIST_A = 10.0
ILT_RESIST_Tr = 0.5
ILT_CONVERGENCE_TOL = 1e-6
ILT_SVD_K = 10



# 文件路径
INITIAL_MASK_PATH = "../The_cross-transfer_coefficient/data/input/t.png"
TARGET_IMAGE_PATH = "../The_cross-transfer_coefficient/data/input/t.png"
OUTPUT_MASK_PATH = "../The_cross-transfer_coefficient/data/output/optimized_mask_t.png"
RESULTS_IMAGE_PATH = "../The_cross-transfer_coefficient/data/output/results_comparison_t.png"
FITNESS_PLOT_PATH = "../The_cross-transfer_coefficient/data/output/fitness_evolution_t.png"
SAVE_PATH="../The_cross-transfer_coefficient/data/output/save.png"

# 可视化参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'
