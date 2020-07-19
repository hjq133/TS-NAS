import numpy as np
import os
import genotypes
from genotypes import Genotype
from utils import create_exp_dir


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
        self.sap_time = [np.zeros([self.num_op * (i + 1)]) for i in range(num_node)]  # 记录sample次数
        self.reward = [np.zeros(self.num_op * (i + 1)) for i in range(num_node)]

    def overwrite_edge_weight(self, edge_reward):
        """
        Overwrites the existing edge rewards with specified values.
        """
        for i in range(self.num_node):
            for j in range((i + 1) * self.num_op):
                self.reward[i][j] = edge_reward[i][j]  # 元祖的赋值是deepcopy

    def get_optimal_network(self):  # 得到权重之和最大的network
        """
        Finds the shortest path through the binomial tree.
        Returns:
          network - list of nodes traversed in order.
        """
        prev_node = []
        activation = []
        for i in range(self.num_node):
            index = np.argmax(self.reward[i])
            cur_group = int(index / self.num_op)
            prev_node.append(cur_group)  # 该点的前置节点
            activation.append(index % self.num_op)
            self.sap_time[i][index] += 1  # 记录采样次数
        # Updating the optimal reward
        return prev_node, activation

    def warm_up_network(self):  # warm up random sample
        prev_node = []
        activation = []
        for i in range(self.num_node):
            val = np.min(self.sap_time[i])

            indices = []  # 保证random choice
            for j in range(len(self.sap_time[i])):
                if self.sap_time[i][j] == val:
                    indices.append(j)
            index = np.random.choice(indices)
            
            cur_group = int(index / self.num_op)
            prev_node.append(cur_group)  # 该点的前置节点
            activation.append(index % self.num_op)
            self.sap_time[i][index] += 1
        return prev_node, activation

    def save_table(self, path):
        create_exp_dir(os.path.dirname(path))
        np.savez(path, sap_time=self.sap_time)


class BanditTS(object):
    def __init__(self, args):
        super().__init__()
        self.num_node = genotypes.STEPS
        self.func_name = genotypes.PRIMITIVES
        self.num_op = len(self.func_name)
        self.sigma_tilde = args.sigma_tilde
        # Set up the internal environment with arbitrary initial values
        self.cell = NeuralGraph(self.num_node, self.num_op, args.mu0, args.sigma0)

        # Save the posterior for edges as tuple (mean, std) of posterior belief
        self.posterior = [[(args.mu0, args.sigma0) for _ in range(i * self.num_op)] for i in
                          range(1, self.num_node + 1)]  # 初始化每条边的后验分布

    def construct_genotype(self, prev_node, activation):
        recurrent = []
        concat = range(1, self.num_node + 1)
        for _, (node, func_id) in enumerate(zip(prev_node, activation)):
            name = self.get_activation(func_id)
            recurrent.append((name, node))
        genotype = Genotype(recurrent=recurrent, concat=concat)
        return genotype

    def get_posterior_sample(self):
        """Gets a posterior sample for each edge.

        Return:
          edge_reward - dict of dicts edge_reward[start_node][end_node] = reward
        """
        edge = [np.zeros(self.num_op * (i + 1)) for i in range(self.num_node)]  # 随机初始化
        for i in range(self.num_node):
            for j in range((i + 1) * self.num_op):
                mean, std = self.posterior[i][j]
                edge[i][j] = mean + std * np.random.randn()

        return edge

        # Bayesian inference
    def bayes(self, old_mean, old_std, reward):
        old_precision = 1. / (old_std ** 2)
        noise_precision = 1. / (self.sigma_tilde ** 2)
        new_precision = old_precision + noise_precision
        new_mean = (noise_precision * reward + old_precision * old_mean) / new_precision
        new_std = np.sqrt(1. / new_precision)
        return new_mean, new_std

    def update_observation(self, prev, act, reward):
        """Updates observations for binomial bridge.
        原文 4.3 的更新公式
        Args:
          action - path chosen by the agent (not used)
          reward - reward
        """

        for i in range(self.num_node):  # reward记录的只是图中一部分的边的reward
            cur_action = prev[i] * self.num_op + act[i]
            old_mean, old_std = self.posterior[i][cur_action]
            new_mean, new_std = self.bayes(old_mean, old_std, reward)
            # update the posterior in place
            self.posterior[i][cur_action] = (new_mean, new_std)

    def pick_action(self):
        """Greedy shortest path wrt posterior sample."""
        posterior_sample = self.get_posterior_sample()
        self.cell.overwrite_edge_weight(posterior_sample)
        prev_node, activation = self.cell.get_optimal_network()
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

    def warm_up_sample(self):  # 根据sample次数进行sample, 保证每条edge都被sample到了
        prev_node, activation = self.cell.warm_up_network()
        return prev_node, activation
