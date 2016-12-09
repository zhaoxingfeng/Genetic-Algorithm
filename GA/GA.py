# coding: utf-8
"""
作者：zhaoxingfeng	日期：2016.12.08
功能：遗传算法，Genetic Algorithm（GA）
版本：2.0
"""
from __future__ import division
import numpy as np
import random
import math
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


class GA(object):
    def __init__(self, maxiter, sizepop, lenchrom, pc, pm, dim, lb, ub, Fobj):
        """
        maxiter：最大迭代次数
        sizepop：种群数量
        lenchrom：染色体长度
        pc：交叉概率
        pm：变异概率
        dim：变量的维度
        lb：最小取值
        ub：最大取值
        Fobj：价值函数
        """
        self.maxiter = maxiter
        self.sizepop = sizepop
        self.lenchrom = lenchrom
        self.pc = pc
        self.pm = pm
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.Fobj = Fobj

    # 初始化种群：返回一个三维数组，第一维是种子，第二维是变量维度，第三维是编码基因
    def Initialization(self):
        pop = []
        for i in range(self.sizepop):
            temp1 = []
            for j in range(self.dim):
                temp2 = []
                for k in range(self.lenchrom):
                    temp2.append(random.randint(0, 1))
                temp1.append(temp2)
            pop.append(temp1)
        return pop

    # 将二进制转化为十进制
    def b2d(self, pop_binary):
        pop_decimal = []
        for i in range(len(pop_binary)):
            temp1 = []
            for j in range(self.dim):
                temp2 = 0
                for k in range(self.lenchrom):
                    temp2 += pop_binary[i][j][k] * math.pow(2, k)
                temp2 = temp2 * (self.ub[j] - self.lb[j]) / (math.pow(2, self.lenchrom) - 1) + self.lb[j]
                temp1.append(temp2)
            pop_decimal.append(temp1)
        return pop_decimal

    # 轮盘赌模型选择适应值较高的种子
    def Roulette(self, fitness, pop):
        # 适应值按照大小排序
        sorted_index = np.argsort(fitness)
        sorted_fitness, sorted_pop = [], []
        for index in sorted_index:
            sorted_fitness.append(fitness[index])
            sorted_pop.append(pop[index])

        # 生成适应值累加序列
        fitness_sum = sum(sorted_fitness)
        accumulation = [None for col in range(len(sorted_fitness))]
        accumulation[0] = sorted_fitness[0] / fitness_sum
        for i in range(1, len(sorted_fitness)):
            accumulation[i] = accumulation[i - 1] + sorted_fitness[i] / fitness_sum

        # 轮盘赌
        roulette_index = []
        for j in range(len(sorted_fitness)):
            p = random.random()
            for k in range(len(accumulation)):
                if accumulation[k] >= p:
                    roulette_index.append(k)
                    break
        temp1, temp2 = [], []
        for index in roulette_index:
            temp1.append(sorted_fitness[index])
            temp2.append(sorted_pop[index])
        newpop = [[x, y] for x, y in zip(temp1, temp2)]
        newpop.sort()
        newpop_fitness = [newpop[i][0] for i in range(len(sorted_fitness))]
        newpop_pop = [newpop[i][1] for i in range(len(sorted_fitness))]
        return newpop_fitness, newpop_pop

    # 交叉繁殖：针对每一个种子，随机选取另一个种子与之交叉。
    # 随机取种子基因上的两个位置点，然后互换两点之间的部分
    def Crossover(self, pop):
        newpop = []
        for i in range(len(pop)):
            if random.random() < self.pc:
                # 选择另一个种子
                j = i
                while j == i:
                    j = random.randint(0, len(pop) - 1)
                cpoint1 = random.randint(1, self.lenchrom - 1)
                cpoint2 = cpoint1
                while cpoint2 == cpoint1:
                    cpoint2 = random.randint(1, self.lenchrom - 1)
                cpoint1, cpoint2 = min(cpoint1, cpoint2), max(cpoint1, cpoint2)
                newpop1, newpop2 = [], []
                for k in range(self.dim):
                    temp1, temp2 = [], []
                    temp1.extend(pop[i][k][0:cpoint1])
                    temp1.extend(pop[j][k][cpoint1:cpoint2])
                    temp1.extend(pop[i][k][cpoint2:])
                    temp2.extend(pop[j][k][0:cpoint1])
                    temp2.extend(pop[i][k][cpoint1:cpoint2])
                    temp2.extend(pop[j][k][cpoint2:])
                    newpop1.append(temp1)
                    newpop2.append(temp2)
                newpop.extend([newpop1, newpop2])
        return newpop

    # 变异：针对每一个种子的每一个维度，进行概率变异，变异基因为一位
    def Mutation(self, pop):
        newpop = copy.deepcopy(pop)
        for i in range(len(pop)):
            for j in range(self.dim):
                if random.random() < self.pm:
                    mpoint = random.randint(0, self.lenchrom - 1)
                    newpop[i][j][mpoint] = 1 - newpop[i][j][mpoint]
        return newpop

    # 绘制迭代-误差图
    def Ploterro(self, Convergence_curve):
        mpl.rcParams['font.sans-serif'] = ['Courier New']
        mpl.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(10, 6))
        x = [i for i in range(len(Convergence_curve))]
        plt.plot(x, Convergence_curve, 'r-', linewidth=1.5, markersize=5)
        plt.xlabel(u'Iter', fontsize=18)
        plt.ylabel(u'Best score', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(0, )
        plt.grid(True)
        plt.show()

    def Run(self):
        pop = self.Initialization()
        errolist = []
        for Current_iter in range(self.maxiter):
            print("Iter = " + str(Current_iter))
            pop1 = self.Crossover(pop)
            pop2 = self.Mutation(pop1)
            pop3 = self.b2d(pop2)
            fitness = []
            for j in range(len(pop2)):
                fitness.append(self.Fobj(pop3[j]))
            sorted_fitness, sorted_pop = self.Roulette(fitness, pop2)
            best_fitness = sorted_fitness[-1]
            best_pos = self.b2d([sorted_pop[-1]])[0]
            pop = sorted_pop[-1:-(self.sizepop + 1):-1]
            errolist.append(1 / best_fitness)
            if 1 / best_fitness < 0.0001:
                print("Best_score = " + str(round(1 / best_fitness, 4)))
                print("Best_pos = " + str([round(a, 4) for a in best_pos]))
                break
        return best_fitness, best_pos, errolist


if __name__ == "__main__":
    # 价值函数，求函数最小值点 -> [1, -1, 0, 0]
    def Fobj(factor):
        cost = (factor[0] - 1) ** 2 + (factor[1] + 1) ** 2 + factor[2] ** 2 + factor[3] ** 2
        return 1 / cost
    starttime = time.time()
    a = GA(100, 50, 10, 0.8, 0.01, 4, [-1, -1, -1, -1], [1, 1, 1, 1], Fobj)
    Best_score, Best_pos, errolist = a.Run()
    endtime = time.time()
    print("Runtime = " + str(endtime - starttime))
    a.Ploterro(errolist)
