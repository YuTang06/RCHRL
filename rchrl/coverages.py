import sys

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq
import random
import gc


class Coverage():
    def __init__(self, x, g, sg, acg, idx, coverage_k, filter_factor_theta, device, no_noif, no_simf):
        super().__init__()
        self.sg_ori = sg
        self.idx_ori = idx

        self.x_ori = x
        self.g_ori = g
        self.acg_ori = acg

        self.k = coverage_k
        self.filter_factor_theta = filter_factor_theta

        self.addl = 10e7
        self.sg = []
        self.x, self.g, self.acg = [], [], []
        self.flag = []
        self.chosen_sg = []
        self.chosen_x, self.chosen_g, self.chosen_acg = [], [], []

        self.classify()

        self.distance, self.smallestk_idx, self.k_dist, self.kth_point_idx= self.k_distance_with_idx()


        if not no_noif and not no_simf:
            self.demon_filter()
            self.cov_simi_cal()
        elif not no_noif and no_simf:
            print("Only demonstration noise subgoal filtering now!")
            self.demon_filter()
        elif not no_simf and no_noif:
            print("Only demonstration similar subgoal filtering now!")
            self.cov_simi_cal()
        else:
            print("There's no demonstration subgoal filtering.")
            for i in range(len(self.sg)):
                for j in range(len(self.sg[i])):
                    chosen_sg_tem = self.sg[i][j]
                    chosen_x_tem = self.x[i][j]
                    chosen_g_tem = self.g[i][j]
                    chosen_acg_tem = self.acg[i][j]
                    self.chosen_sg.append(np.array(chosen_sg_tem))
                    self.chosen_x.append(np.array(chosen_x_tem))
                    self.chosen_g.append(np.array(chosen_g_tem))
                    self.chosen_acg.append(np.array(chosen_acg_tem))
            print('{} demonstration subgoals have been kept.'.format(len(self.chosen_sg)))

        gc.collect()
        torch.cuda.empty_cache()

        return None

    def check_sg(self):
        size = 0
        self.del_num = []
        self.diff_num = []
        self.diff_list = []
        for i in range(len(self.sg)):
            if len(self.sg[i]) > size:
                size = len(self.sg[i])

        for i in range(len(self.sg)):
            if len(self.sg[i]) < size:
                diff = size - len(self.sg[i])
                # fill
                if diff < len(self.sg[i]):
                    sample_list = [i for i in range(len(self.sg[i]))]
                    sample_list = random.sample(sample_list, diff)
                    random_data = self.sg[i][sample_list]
                    self.sg[i] = np.concatenate((self.sg[i], random_data))
                    # for j in range(diff):
                        # self.flag[i][size-j-1] = -1
                    self.diff_num.append(i)
                    self.diff_list.append(diff)
                # delete
                else:
                    self.del_num.append(i)

        del_num = self.del_num
        for i in range(len(self.del_num)):
            del_flag = del_num[i]
            self.sg = np.delete(self.sg, del_flag, axis=None)
            # self.flag = np.delete(self.flag, del_flag, axis=None)
            del_num = [item - 1 for item in del_num]

    def init_flag(self):
        self.flag = [[0 for j in range(len(self.sg[0]))] for i in range(len(self.sg))]

        diff_sq = []
        for del_i in range(len(self.del_num)):
            a = 0
            for diff_i in range(len(self.diff_num)):
                if self.del_num[del_i] < self.diff_num[diff_i]:
                    a += 1
            diff_sq.append(a)
        for i in diff_sq:
            self.diff_num[-i:] = [item - 1 for item in self.diff_num[-i:]]
        for i in range(len(self.diff_num)):
            self.flag[self.diff_num[i]][-self.diff_list[i]:] = [item - 1 for item in self.flag[self.diff_num[i]][-self.diff_list[i]:]]

    def classify(self):
        # to classify subgoals belonging to the same traj
        for e_idx in set(self.idx_ori):
            sg_tem = []
            x_tem, g_tem, acg_tem = [], [], []
            for i in range(len(self.sg_ori)):
                if self.idx_ori[i] == e_idx:
                    sg_tem.append(self.sg_ori[i])
                    x_tem.append(self.x_ori[i])
                    g_tem.append(self.g_ori[i])
                    acg_tem.append(self.acg_ori[i])

            self.sg.append(np.array(sg_tem))
            self.x.append(np.array(x_tem))
            self.g.append(np.array(g_tem))
            self.acg.append(np.array(acg_tem))

        self.check_sg()
        self.init_flag()

    def k_distance_with_idx(self):
        dist = []
        smallestk_idx = []
        k_dist = []
        kth_point_idx = []
        for i in range(len(self.sg)):
            dist_episode = []
            smallestk_idx_episode = []
            k_dist_episode = []
            kth_point_idx_episode = []
            for j in range(len(self.sg[i])):
                dist_episode_ele, smallestk_idx_episode_ele, k_dist_ele, kth_point_idx_ele= self.calcu_dist(i, self.sg[i][j], self.sg)
                # dist_episode_ele, smallestk_idx_episode_ele, k_dist_ele= self.calcu_dist(i, self.sg[i][j], np.delete(self.sg, i, axis=0))
                dist_episode.append(np.array(dist_episode_ele))
                smallestk_idx_episode.append(smallestk_idx_episode_ele)
                k_dist_episode.append(k_dist_ele)
                kth_point_idx_episode.append(kth_point_idx_ele)
            dist.append(np.array(dist_episode))
            smallestk_idx.append(smallestk_idx_episode)
            k_dist.append(k_dist_episode)
            kth_point_idx.append(kth_point_idx_episode)

        return np.array(dist), smallestk_idx, k_dist, kth_point_idx

    def calcu_dist(self,ep_idx, x, y):
        dist = []
        smallest = []
        smallest_index = []
        # np.delete(self.sg, ep_idx, axis=0)

        for r in range(len(y)):
            dist_tem = np.sqrt(np.sum(np.asarray(x - y[r]) ** 2, axis=1))
            if r == ep_idx:
                dist_tem += self.addl
            smallest_tem = heapq.nsmallest(self.k, dist_tem)

            dist.append(dist_tem)
            smallest.extend(smallest_tem)

        for i in range(len(heapq.nsmallest(self.k , smallest))):
            smallest_index_ele = np.where(dist == heapq.nsmallest(self.k , smallest)[i])
            if len(smallest_index_ele[0]) != 1:
                smallest_index_ele = smallest_index_ele[0]
            # smallest5_index = list(map(dist.index, heapq.nsmallest(self.k , smallest)[i]))
            smallest_index_list= [smallest_index_ele[0].squeeze().tolist(),smallest_index_ele[1].squeeze().tolist()]

            smallest_index.append(list(smallest_index_list))

        k_dist = heapq.nlargest(1, heapq.nsmallest(self.k , smallest))
        kth_point_idx = np.where(dist == k_dist[0])
        kth_point_idx_list = [kth_point_idx[0].squeeze().tolist(), kth_point_idx[1].squeeze().tolist()]

        return np.array(dist),smallest_index, k_dist, kth_point_idx_list

    def demon_filter(self,factor_filted = True):
        # self.demon_filter(self.smallestk_idx, self.k_dist)

        self.reach_cov = np.pi * np.power(self.k_dist, 2)
        self.r_cov_dens = self.reach_cov / self.k
        x = []
        y = []
        outlier = []

        if factor_filted:
            for i in range(len(self.sg)):
                sg_k_cov_dens = 0
                for j in range(len(self.sg[i])):
                    sg_cov_dens = self.r_cov_dens[i][j]
                    k_idx = self.smallestk_idx[i][j]
                    for sn in range(self.k):
                        sg_k_cov_dens += self.r_cov_dens[k_idx[sn][0]][k_idx[sn][1]]

                    sg_k_cov_dens /= self.k
                    filter_factor = sg_cov_dens / sg_k_cov_dens
                    if filter_factor > self.filter_factor_theta:
                        outlier.append(self.sg[i][j])
                        self.flag[i][j] = -2
        else:
            r_cov_dens_ave = np.mean(self.r_cov_dens)
            for i in range(len(self.sg)):
                for j in range(len(self.sg[i])):
                    if self.r_cov_dens[i][j] > r_cov_dens_ave:
                        outlier.append(self.sg[i][j])
                        self.flag[i][j] = -2
        print('Imperfect demonstration subgoals', self.flag)

        return 0

    def cov_simi_cal(self):
        # self.distance, self.smallestk_idx, self.k_dist, self.kth_point_idx = self.k_distance_with_idx()
        # self.reach_cov = np.pi * np.power(self.k_dist, 2)
        # self.r_cov_dens = self.reach_cov  / self.k

        tem_x = 0
        tem_y = 0
        times = 0

        while len(np.argwhere(np.array(self.flag) == [0])) is not 0:

            # if core sg hoave been already flaged, the coverge circle ends
            while True:
                if isinstance(self.kth_point_idx[tem_x][tem_y][0], int) == False:
                    for i in range(1, len(self.kth_point_idx[tem_x][tem_y][0])):
                        self.flag[self.kth_point_idx[tem_x][tem_y][0][i]][self.kth_point_idx[tem_x][tem_y][1][i]] = -1
                    self.kth_point_idx[tem_x][tem_y][0] = self.kth_point_idx[tem_x][tem_y][0][0]
                    self.kth_point_idx[tem_x][tem_y][1] = self.kth_point_idx[tem_x][tem_y][1][0]

                if self.flag[self.kth_point_idx[tem_x][tem_y][0]][self.kth_point_idx[tem_x][tem_y][1]] in [-2, -1, 1]:
                    self.flag[tem_x][tem_y] = -1
                    if len(np.argwhere(np.array(self.flag) == [0])) is not 0:
                        random.seed(2)
                        new_idx = np.argwhere(np.array(self.flag) == [0])[random.randint(0,len(np.argwhere(np.array(self.flag) == [0]))-1)]
                        tem_x = new_idx[0].squeeze().tolist()
                        tem_y = new_idx[1].squeeze().tolist()
                    else:
                        print('Critical demonstration subgoal chosen is already done!')
                        break
                else:
                    x = tem_x
                    y = tem_y
                    break

            self.flag[x][y] = 1
            for i in range(len(self.smallestk_idx[x][y])):
                id = self.smallestk_idx[x][y][i]
                if self.flag[id[0]][id[1]] == 0:
                    self.flag[id[0]][id[1]] = -1

            # all delete
            for i in range(len(self.k_dist[x])):
                if self.distance[x][y][x][i] - self.addl < self.k_dist[x][y] and self.flag[x][i] == 0:
                    self.flag[x][i] = -1

            self.flag[self.kth_point_idx[x][y][0]][self.kth_point_idx[x][y][1]] = 1

            # next_core_idx
            tem_x = self.kth_point_idx[x][y][0]
            tem_y = self.kth_point_idx[x][y][1]
            times += 1

        for i in range(len(np.argwhere(np.array(self.flag) == [1]))):
            chosen_location = np.argwhere(np.array(self.flag) == [1])[i]

            if chosen_location[1] < len(self.x[chosen_location[0]]):
                chosen_sg_tem = self.sg[chosen_location[0]][chosen_location[1]]
                chosen_x_tem = self.x[chosen_location[0]][chosen_location[1]]
                chosen_g_tem = self.g[chosen_location[0]][chosen_location[1]]
                chosen_acg_tem = self.acg[chosen_location[0]][chosen_location[1]]

                self.chosen_sg.append(np.array(chosen_sg_tem))
                self.chosen_x.append(np.array(chosen_x_tem))

                self.chosen_g.append(np.array(chosen_g_tem))
                self.chosen_acg.append(np.array(chosen_acg_tem))

        # print('chosen sg',self.chosen_sg)
        print('{} critical demonstration subgoals have been selected.'.format(len(self.chosen_sg)))


    def SeqDemonRead(self):
        ep_obs_seq_demon, ep_ac_seq_demon = [], []
        for i in range(len(self.chosen_sg)):
            ep_obs_seq_demon.append(self.chosen_x[i])
            ep_ac_seq_demon.append(self.chosen_acg[i])
        gc.collect()
        torch.cuda.empty_cache()
        return ep_obs_seq_demon, ep_ac_seq_demon

    def BufferDemonRead(self, manager_buffer, controller_buffer):
        for i in range(len(self.chosen_sg)):
            manager_buffer.add((self.chosen_x[i], None, self.chosen_acg[i], None, self.chosen_g[i], self.chosen_sg[i], 0, False, [self.chosen_x[i]], [], [self.chosen_acg[i]]))
            controller_buffer.add((self.chosen_x[i], None, self.chosen_acg[i], None, self.chosen_sg[i], None, 0, 0, [], [], []))
        gc.collect()
        torch.cuda.empty_cache()
        return manager_buffer, controller_buffer