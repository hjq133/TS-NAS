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
        self.norm_sap_time = [np.zeros([i * num_op]) for i in range(1, num_node + 1)]  # 记录sample次数
        self.redu_sap_time = [np.zeros([i * num_op]) for i in range(1, num_node + 1)]  # 记录sample次数
        self.norm_reward = [np.zeros(4 * i) for i in range(1, num_node + 1)]
        self.redu_reward = [np.zeros(4 * i) for i in range(1, num_node + 1)]

    def overwrite_edge_weight(self, redu_edge, norm_edge):
        """
        Overwrites the existing edge rewards with specified values.
        """
        for i in range(self.num_node):
            for j in range((i + 1) * self.num_op):
                self.norm_reward[i][j] = norm_edge[i][j]  # 元祖的赋值是deepcopy
                self.redu_reward[i][j] = redu_edge[i][j]

    def get_optimal_network(self):  # 得到权重之和最大的network
        """
        Finds the shortest path through the binomial tree.
        Returns:
          network - list of nodes traversed in order.
        """
        norm_prev = []
        norm_act = []
        redu_prev = []
        redu_act = []
        for i in range(self.num_node):
            norm_index = np.argmax(self.norm_reward[i])
            redu_index = np.argmax(self.redu_reward[i])
            self.norm_sap_time[i][norm_index] += 1  # 记录采样次数
            self.redu_sap_time[i][redu_index] += 1
            norm_prev.append(int(norm_index / self.num_op))  # 该点的前置节点
            norm_act.append(norm_index % self.num_op)
            redu_prev.append(int(redu_index / self.num_op))
            redu_act.append(redu_index % self.num_op)
        return norm_prev, norm_act, redu_prev, redu_act

    def save_table(self, path):
        create_exp_dir(os.path.dirname(path))
        np.savez(path, norm_sap_time=self.norm_sap_time, redu_sap_time=self.redu_sap_time)


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
        self.posterior_norm = [[(args.mu0, args.sigma0) for _ in range(i * self.num_op)] for i in
                               range(2, self.num_node + 2)]  # 初始化每条边的后验分布
        self.posterior_redu = [[(args.mu0, args.sigma0) for _ in range(i * self.num_op)] for i in
                               range(2, self.num_node + 2)]  # 初始化每条边的后验分布

    def construct_genotype(self, norm_prev, norm_act, redu_prev, redu_act):
        normal = []
        reduce = []
        concat = range(2, self.num_node + 2)  # TODO: should be more flexible
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
          edge_length - dict of dicts edge_length[start_node][end_node] = distance
        """
        redu_edge = [np.zeros(4 * i) for i in range(1, self.num_node + 1)]
        norm_edge = [np.zeros(4 * i) for i in range(1, self.num_node + 1)]  # 随机初始化
        for i in range(self.num_node):
            for j in range((i + 1) * self.num_op):
                mean, std = self.posterior_norm[i][j]
                norm_edge[i][j] = mean + std * np.random.randn()
                mean, std = self.posterior_redu[i][j]
                redu_edge[i][j] = mean + std * np.random.randn()
        return redu_edge, norm_edge

    def bayes(self, old_mean, old_std, reward):
        old_precision = 1. / (old_std ** 2)
        noise_precision = 1. / (self.sigma_tilde ** 2)
        new_precision = old_precision + noise_precision
        new_mean = (noise_precision * (reward + 0.5 / noise_precision) + old_precision * old_mean) / new_precision
        new_std = np.sqrt(1. / new_precision)
        return new_mean, new_std

    def update_observation(self, norm_prev, norm_act, redu_prev, redu_act, reward):
        """Updates observations for binomial bridge.
        Args:
          action - path chosen by the agent (not used)
          reward - reward
        """
        # update norm cell
        for i in range(self.num_node):  # reward记录的只是图中一部分的边的reward
            cur_action = norm_prev[i] * self.num_op + norm_act[i]
            old_mean, old_std = self.posterior_norm[i][cur_action]
            # convert std into precision for easier algebra
            new_mean, new_std = self.bayes(old_mean, old_std, reward)
            # update the posterior in place
            self.posterior_norm[i][cur_action] = (new_mean, new_std)

        # update reduction cell
        for i in range(self.num_node):  # reward记录的只是图中一部分的边的reward
            cur_action = redu_prev[i] * self.num_op + redu_act[i]
            old_mean, old_std = self.posterior_redu[i][cur_action]
            # convert std into precision for easier algebra
            new_mean, new_std = self.bayes(old_mean, old_std, reward)
            # update the posterior in place
            self.posterior_redu[i][cur_action] = (new_mean, new_std)

    def pick_action(self):
        """Greedy shortest path wrt posterior sample."""
        redu_sample, norm_sample = self.get_posterior_sample()
        self.internal_env.overwrite_edge_weight(redu_sample, norm_sample)
        norm_prev, norm_act, redu_prev, redu_act = self.internal_env.get_optimal_network()
        return norm_prev, norm_act, redu_prev, redu_act

    # def derive_sample(self):  # TODO 选择最好的那些
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
