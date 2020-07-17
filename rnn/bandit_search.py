import numpy as np
import random
import collections
import itertools
import os
import genotypes
from genotypes import Genotype
from utils import make_dirs


class NeuralGraph(object):

    def __init__(self, num_node, num_op, mu0, sigma0):
        """An env for graph bandits.

        Args:
          num_node - number of neural nodes
          num_op - number of operation for each node
          mu0 - prior mean
          sigma0 - prior stddev
        """
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.num_node = num_node
        self.num_op = num_op
        self.sap_time = [np.zeros([i * num_op]) for i in range(1, num_node + 1)]  # 记录sample次数
        self.reward = [np.zeros(4 * i) for i in range(1, num_node + 1)]

    def overwrite_edge_weight(self, edge_reward):
        """Overwrites the existing edge weights with specified values.

        Args:
          edge_length - dict of dicts edge_weight[i][j] = weight
        """
        for i in range(self.num_node):
            for j in range((i + 1) * self.num_op):
                self.reward[i][j] = edge_reward[i][j]  # 元祖的赋值是deepcopy

    def get_optimal_network(self):  # 得到权重之和最大的network
        """Finds the shortest path through the binomial tree.

        Returns:
          network - list of nodes traversed in order.
        """
        prev_node = []
        activation = []
        for i in range(self.num_node):
            s_index = np.argmax(self.reward[i])
            self.sap_time[i][s_index] += 1  # 记录采样次数
            prev_node.append(int(s_index / self.num_op))  # 该点的前置节点
            activation.append(s_index % self.num_op)

        # Updating the optimal reward
        return prev_node, activation

    def save_table(self, path):
        make_dirs(os.path.dirname(path))
        np.savez(path, sap_time=self.sap_time)


class BanditTS(object):
    def __init__(self, args):
        super().__init__()
        self.num_node = genotypes.STEPS
        self.func_name = genotypes.PRIMITIVES
        self.num_op = len(self.func_name)
        self.sigma_tilde = args.sigma_tilde

        # Set up the internal environment with arbitrary initial values
        self.internal_env = NeuralGraph(self.num_node, self.num_op, args.mu0, args.sigma0)

        # Save the posterior for edges as tuple (mean, std) of posterior belief
        self.posterior = [[(args.mu0, args.sigma0) for _ in range(i * self.num_op)] for i in
                          range(1, self.num_node + 1)]  # 初始化每条边的后验分布

    def construct_genotype(self, prev_node, activation):
        recurrent = []
        for _, (node, func_id) in enumerate(zip(prev_node, activation)):
            name = self.get_activation(func_id)
            recurrent.append((name, node))
        genotype = Genotype(recurrent=recurrent, concat=range(1, self.num_node + 1))
        return genotype

    def get_posterior_sample(self):
        """Gets a posterior sample for each edge.

        Return:
          edge_length - dict of dicts edge_length[start_node][end_node] = distance
        """
        edge_reward = [np.zeros(4 * i) for i in range(1, self.num_node + 1)]  # 随机初始化

        for i in range(self.num_node):
            for j in range((i + 1) * self.num_op):
                mean, std = self.posterior[i][j]
                edge_reward[i][j] = mean + std * np.random.randn()

        return edge_reward

    def update_observation(self, prev, act, data):
        """Updates observations for binomial bridge.
        原文 4.3 的更新公式
        Args:
          action - path chosen by the agent (not used)
          reward - reward
        """

        for i in range(self.num_op):  # reward记录的只是图中一部分的边的reward
            cur_action = prev[i] * self.num_op + act[i]
            old_mean, old_std = self.posterior[i][cur_action]

            # convert std into precision for easier algebra
            old_precision = 1. / (old_std ** 2)
            noise_precision = 1. / (self.sigma_tilde ** 2)
            new_precision = old_precision + noise_precision

            new_mean = (noise_precision * data + old_precision * old_mean) / new_precision
            new_std = np.sqrt(1. / new_precision)

            # update the posterior in place
            self.posterior[i][cur_action] = (new_mean, new_std)

    def pick_action(self):
        """Greedy shortest path wrt posterior sample."""
        posterior_sample = self.get_posterior_sample()
        self.internal_env.overwrite_edge_weight(posterior_sample)
        prev_node, activation = self.internal_env.get_optimal_network()
        return prev_node, activation

    def get_activation(self, act):
        if act == 0:
            return 'tanh'
        elif act == 1:
            return 'relu'
        elif act == 2:
            return 'identity'
        elif act == 3:
            return 'sigmoid'
        else:
            raise NotImplementedError

    def derive_sample(self):  # TODO 选择最好的那些
        activation = []
        prev_node = []
        for i in range(self.num_node):
            mean = 0
            id = 0
            for index in range((i + 1) * self.num_op):
                if self.posterior[i][index][0] > mean:
                    mean = self.posterior[i][index][0]
                    id = index
            prev_node.append(int(id / self.num_op))
            activation.append(id % self.num_op)
        return prev_node, activation
