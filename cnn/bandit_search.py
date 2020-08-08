import numpy as np
import os
import genotypes
from genotypes import Genotype, PRIMITIVES
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
        self.sap_time = [np.zeros([self.num_op * (i + 2)]) for i in range(self.num_node)]  # 记录sample次数
        self.reward = [np.zeros(self.num_op * (i + 2)) for i in range(self.num_node)]
        self.pruned_index = [[] for _ in range(self.num_node)]
        self.sample_index = [np.arange(self.num_op * (i + 2)) for i in range(self.num_node)]

    def overwrite_edge_weight(self, edge_reward):
        """
        Overwrites the existing edge rewards with specified values.
        """
        for i in range(self.num_node):
            for j in range((i + 2) * self.num_op):
                self.reward[i][j] = edge_reward[i][j]  # 元祖的赋值是deepcopy

    def get_optimal_network(self, top_k):  # 得到权重之和最大的network
        """
        Finds the shortest path through the binomial tree.
        Returns:
          network - list of nodes traversed in order.
        """
        prev_node = []
        activation = []
        for i in range(self.num_node):
            for _ in range(top_k):  # 保留topk的边
                index = np.argmax(self.reward[i])
                cur_group = int(index / self.num_op)
                prev_node.append(cur_group)  # 该点的前置节点
                activation.append(index % self.num_op)
                self.sap_time[i][index] += 1  # 记录采样次数
                self.reward[i][cur_group * self.num_op: (cur_group + 1) * self.num_op] = -9999  # 避免重复采样同一个前驱节点
        return prev_node, activation

    def warm_up_network(self, top_k):  # 保留k条前缀边
        prev_node = []
        activation = []
        for i in range(self.num_node):
            for _ in range(top_k):  # 保留topk的边
                index = np.argmin(self.sap_time[i])
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
        self.top_k = args.top_k
        self.num_node = genotypes.STEPS
        self.func_name = genotypes.PRIMITIVES
        self.num_op = len(self.func_name)
        self.sigma_tilde = args.sigma_tilde
        # Set up the internal environment with arbitrary initial values
        self.norm_cell = NeuralGraph(self.num_node, self.num_op, args.mu0, args.sigma0)
        self.redu_cell = NeuralGraph(self.num_node, self.num_op, args.mu0, args.sigma0)
        # Save the posterior for edges as tuple (mean, std) of posterior belief
        self.posterior_norm = [[(args.mu0, args.sigma0) for _ in range(i * self.num_op)] for i in
                               range(2, self.num_node + 2)]  # 初始化每条边的后验分布
        self.posterior_redu = [[(args.mu0, args.sigma0) for _ in range(i * self.num_op)] for i in
                               range(2, self.num_node + 2)]  # 初始化每条边的后验分布

    def construct_genotype(self, norm_prev, norm_act, redu_prev, redu_act):
        normal = []
        reduce = []
        concat = range(2, self.num_node + 2)
        for _, (node, func_id) in enumerate(zip(norm_prev, norm_act)):
            name = PRIMITIVES[func_id]
            normal.append((name, node))
        for _, (node, func_id) in enumerate(zip(redu_prev, redu_act)):
            name = PRIMITIVES[func_id]
            reduce.append((name, node))
        genotype = Genotype(normal=normal, normal_concat=concat, reduce=reduce, reduce_concat=concat)
        return genotype

    def get_posterior_sample(self):
        """Gets a posterior sample for each edge.
        Return:
          edge_reward - dict of dicts edge_reward[start_node][end_node] = reward
        """
        redu_edge = [np.zeros(self.num_op * (i + 2)) for i in range(self.num_node)]
        norm_edge = [np.zeros(self.num_op * (i + 2)) for i in range(self.num_node)]  # 随机初始化
        for i in range(self.num_node):
            for j in range((i + 2) * self.num_op):
                mean, std = self.posterior_norm[i][j]
                norm_edge[i][j] = mean + std * np.random.randn()
                mean, std = self.posterior_redu[i][j]
                redu_edge[i][j] = mean + std * np.random.randn()
        return redu_edge, norm_edge

    # Bayesian inference
    def bayes(self, old_mean, old_std, data):
        old_precision = 1. / (old_std ** 2)
        noise_precision = 1. / (self.sigma_tilde ** 2)
        new_precision = old_precision + noise_precision
        new_mean = (noise_precision * data + old_precision * old_mean) / new_precision
        new_std = np.sqrt(1. / new_precision)
        return new_mean, new_std

    def update_observation(self, norm_prev, norm_act, redu_prev, redu_act, reward):  # TODO how to update ?
        """Updates observations for binomial bridge.
        Args:
          action - path chosen by the agent (not used)
          reward - reward
        """
        # update norm cell
        for i in range(self.num_node):  # reward记录的只是图中一部分的边的reward
            for j in range(self.top_k):  # 保留了top k的边
                cur_action = norm_prev[self.top_k * i + j] * self.num_op + norm_act[self.top_k * i + j]
                old_mean, old_std = self.posterior_norm[i][cur_action]
                # convert std into precision for easier algebra
                new_mean, new_std = self.bayes(old_mean, old_std, reward)
                # update the posterior in place
                self.posterior_norm[i][cur_action] = (new_mean, new_std)

        # update reduction cell
        for i in range(self.num_node):  # reward记录的只是图中一部分的边的reward
            for j in range(self.top_k):
                cur_action = redu_prev[self.top_k * i + j] * self.num_op + redu_act[self.top_k * i + j]
                old_mean, old_std = self.posterior_redu[i][cur_action]
                # convert std into precision for easier algebra
                new_mean, new_std = self.bayes(old_mean, old_std, reward)
                # update the posterior in place
                self.posterior_redu[i][cur_action] = (new_mean, new_std)

    def pick_action(self):
        """Greedy shortest path wrt posterior sample."""
        redu_sample, norm_sample = self.get_posterior_sample()
        self.redu_cell.overwrite_edge_weight(redu_sample)
        self.norm_cell.overwrite_edge_weight(norm_sample)
        norm_prev, norm_act = self.norm_cell.get_optimal_network(self.top_k)
        redu_prev, redu_act = self.redu_cell.get_optimal_network(self.top_k)
        return norm_prev, norm_act, redu_prev, redu_act

    # def derive_sample(self):  # TODO 如何derive
    #     activation = []
    #     prev_node = []
    #     for i in range(self.num_node):
    #         mean = 0
    #         id = 0
    #         for index in range((i + 1) * self.num_op):
    #             if self.posterior[i][index][0] > mean:
    #                 mean = self.posterior[i][index][0]
    #                 id = index
    #         prev_node.append(int(id / self.num_op))
    #         activation.append(id % self.num_op)
    #     return prev_node, activation

    def warm_up_sample(self):  # 根据sample次数进行sample, 保证每条edge都被sample到了
        norm_prev, norm_act = self.norm_cell.warm_up_network(self.top_k)
        redu_prev, redu_act = self.redu_cell.warm_up_network(self.top_k)
        return norm_prev, norm_act, redu_prev, redu_act

    def prune_op(self, index):
        repeat_num = 10
        redu_edge = np.zeros(self.num_op * (index + 2))
        norm_edge = np.zeros(self.num_op * (index + 2))  # 随机初始化
        for j in range((index + 2) * self.num_op):
            mean, std = self.posterior_norm[index][j]
            value = 0
            for i in range(repeat_num):
                value += mean + std * np.random.randn()
            norm_edge[j] = value / repeat_num

            mean, std = self.posterior_redu[index][j]
            value = 0
            for i in range(repeat_num):
                value += mean + std * np.random.randn()
            redu_edge[j] = value / repeat_num
        norm_edge[self.norm_cell.pruned_index[index]] = np.nan
        redu_edge[self.redu_cell.pruned_index[index]] = np.nan
        norm_pruned = np.nanargmin(norm_edge)
        redu_pruned = np.nanargmin(redu_edge)
        self.norm_cell[index].append(norm_pruned)
        self.redu_cell[index].append(redu_pruned)

    def prune_op(self, epoch):
        if epoch in [40, 75, 105, 130, 150, 165, 175]:
            self.prune_op(0)
        if epoch in [80, 130, 170, 200, 220, 240, 250]:
            self.prune_op(1)
        if epoch in [100, 150, 190, 250, 280, 310, 330]:
            self.prune_op(2)
        if epoch in [120, 180, 230, 270, 300, 320, 340]:
            self.prune_op(3)
