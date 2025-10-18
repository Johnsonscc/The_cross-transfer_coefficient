import numpy as np
from deap import base, creator, tools, algorithms
from config.parameters import *
from utils.image_processing import binarize_image
from core.lithography_simulation_source import hopkins_digital_lithography_simulation

# 创建适应度和个体类
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def setup_toolbox(initial_mask_flat, target_image, threshold):
    toolbox = base.Toolbox()

    # 使用原始掩膜和噪声创建个体
    def create_individual_with_noise(initial_mask_flat, noise_scale=NOISE_SCALE):
        noise = np.random.normal(0, noise_scale, size=initial_mask_flat.shape)#生成正态分布的随机噪声
        individual = initial_mask_flat + noise
        return np.clip(individual, 0, 1)#限制数值在[0,1]

    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: create_individual_with_noise(initial_mask_flat))#迭代初始化，个体类型
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)#重复创建，容器类型

    # 评估函数
    def evalPE(individual):
        mask = np.array(individual, dtype=np.float32).reshape((LX, LY))#一维个体转化为二维掩模
        simulated_image = hopkins_digital_lithography_simulation(mask)
        binary_image = binarize_image(simulated_image)
        PE = np.mean((binary_image.astype(np.float32) - target_image.astype(np.float32)) ** 2)#MSE
        return PE,


    toolbox.register("evaluate", evalPE)
    toolbox.register("mate", tools.cxTwoPoint)#两点交叉
    toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)#位翻转变异
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)#锦标赛机制

    return toolbox


def run_genetic_algorithm(toolbox, population_size=POPULATION_SIZE, generations=GENERATIONS,
                          cxpb=CXPB, mutpb=MUTPB):
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)


    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                   ngen=generations, stats=stats,
                                   halloffame=hof, verbose=True)

    return pop, hof, log