import numpy as np
import genotypes
import argparse
from bandit_search import BanditTS
from genotypes import Genotype, PRIMITIVES

parser = argparse.ArgumentParser("cifar")
# thompson sampling parameters
parser.add_argument('--reward_c', type=int, default=3)  # reward 基数  # TODO 每条边所获得的reward需要与训练次数t作比较
parser.add_argument('--mu0', type=int, default=1)  # 高斯分布的均值  TODO 初值是不是很重要
parser.add_argument('--sigma0', type=int, default=1)  # 高斯分布的方差  TODO 初值是不是很重要
parser.add_argument('--sigma_tilde', type=int, default=1)  # 方差 assume为1
parser.add_argument('--top_k', type=int, default=2)  # 保留top k的边
args = parser.parse_args()


def test_warm_up_network():
    epoch = 40
    bandit = BanditTS(args)
    for i in range(epoch):  # num node
        bandit.warm_up_sample()
    for i in range(genotypes.STEPS):
        print(bandit.norm_cell.sap_time[i])
        print('-' * 89)


test_warm_up_network()
