import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class ArmDrawerHDDataset(Dataset):
    def __init__(self, device, file_path, transform=None):
        self.data_frame = pd.read_csv(file_path, header=None)
        self.transform = transform
        self.device = device

    def __getitem__(self, idx):
        e_idx = float(self.data_frame.iloc[idx, 0][1:])

        X = [float(self.data_frame.iloc[idx, 1][2:])]
        for i in range(2, 39):
            X.append(float(self.data_frame.iloc[idx, i]))
        X.append(float(self.data_frame.iloc[idx, 39][:-1]))

        G = [float(self.data_frame.iloc[idx, 40][2:])]
        G.append(float(self.data_frame.iloc[idx, 41]))
        G.append(float(self.data_frame.iloc[idx, 42][:-1]))

        SG = [float(self.data_frame.iloc[idx, 43][2:])]
        for i in range(44, 48):
            SG.append(float(self.data_frame.iloc[idx, i]))
        SG.append(float(self.data_frame.iloc[idx, 48][:-1]))

        ASG = [float(self.data_frame.iloc[idx, 49][2:])]
        ASG.append(float(self.data_frame.iloc[idx, 50]))
        ASG.append(float(self.data_frame.iloc[idx, 51][:-2]))

        return X, G, SG, ASG, e_idx

    def __len__(self):
        return len(self.data_frame)


class ArmDoor_ReachHDDataset(Dataset):
    def __init__(self, device, file_path, transform=None):
        self.data_frame = pd.read_csv(file_path, header=None)
        self.transform = transform
        self.device = device

    def __getitem__(self, idx):
        e_idx = float(self.data_frame.iloc[idx, 0][1:])

        X = [float(self.data_frame.iloc[idx, 1][2:])]
        for i in range(2, 39):
            X.append(float(self.data_frame.iloc[idx, i]))
        X.append(float(self.data_frame.iloc[idx, 39][:-1]))

        G = [float(self.data_frame.iloc[idx, 40][2:])]
        G.append(float(self.data_frame.iloc[idx, 41]))
        G.append(float(self.data_frame.iloc[idx, 42][:-1]))

        SG = [float(self.data_frame.iloc[idx, 43][2:])]
        SG.append(float(self.data_frame.iloc[idx, 44]))
        SG.append(float(self.data_frame.iloc[idx, 45][:-1]))

        ASG = [float(self.data_frame.iloc[idx, 46][2:])]
        ASG.append(float(self.data_frame.iloc[idx, 47]))
        ASG.append(float(self.data_frame.iloc[idx, 48][:-2]))

        return X, G, SG, ASG, e_idx

    def __len__(self):
        return len(self.data_frame)

class AntMazeHDDataset(Dataset):
    # episode_num, state, goal, subgoal, achieved_goal
    def __init__(self, device, file_path, transform=None):
        self.data_frame = pd.read_csv(file_path, header=None)
        self.transform = transform
        self.device = device

    def __getitem__(self, idx):
        e_idx = float(self.data_frame.iloc[idx, 0][1:])

        X = [float(self.data_frame.iloc[idx, 1][2:])]
        for i in range(2, 30):
            X.append(float(self.data_frame.iloc[idx, i]))
        X.append(float(self.data_frame.iloc[idx, 30][:-1]))

        G = [float(self.data_frame.iloc[idx, 31][2:])]
        G.append(float(self.data_frame.iloc[idx, 32][:-1]))

        SG = [float(self.data_frame.iloc[idx, 33][2:])]
        SG.append(float(self.data_frame.iloc[idx, 34][:-1]))

        ASG = [float(self.data_frame.iloc[idx, 35][2:])]
        ASG.append(float(self.data_frame.iloc[idx, 36][:-2]))

        return X, G, SG, ASG, e_idx

    def __len__(self):
        return len(self.data_frame)

class HighDemon():
    def __init__(self, filename, device):
        super().__init__()
        self.device = device
        self.filename = filename

        if 'AntMaze' in filename:
            self.read_dataset = AntMazeHDDataset(self.device, self.filename)
        elif 'reach' in filename:
            self.read_dataset = ArmDoor_ReachHDDataset(self.device, self.filename)
        elif 'door' in filename:
            self.read_dataset = ArmDoor_ReachHDDataset(self.device, self.filename)
        elif 'drawer' in filename:# drawer-open
            self.read_dataset = ArmDrawerHDDataset(self.device, self.filename)
        else:
            assert False, 'Unknown env'

        self.x, self.g, self.sg, self.acg, self.idx  = [], [] , [], [], []


    def HighDemonRead(self):
        print("****************************************************************")
        print("Obtaining High-level Demon from {} ".format(self.filename))
        print("****************************************************************")
        x, g, sg, acg, idx = [],[],[],[],[]

        for i in range(len(self.read_dataset)):
            x_tem,g_tem,sg_tem,acg_tem,idx_tem = self.read_dataset.__getitem__(i)
            x.append(x_tem)
            g.append(g_tem)
            sg.append(sg_tem)
            acg.append(acg_tem)
            idx.append(idx_tem)
        self.x = np.array(x)
        self.g = np.array(g)
        self.sg = np.array(sg)
        self.acg = np.array(acg)
        self.idx = np.array(idx)

        return self.x, self.g, self.sg, self.acg, self.idx

    def SeqDemonRead(self, x, acg):
        ep_obs_seq_demon, ep_ac_seq_demon = [], []
        for i in range(len(x)):
            ep_obs_seq_demon.append(x[i])
            ep_ac_seq_demon.append(acg[i])
        return ep_obs_seq_demon , ep_ac_seq_demon

    def BufferDemonRead(self, manager_buffer, controller_buffer, x, g, sg, acg):
        for i in range(len(x)):
            manager_buffer.add((x[i], None, acg[i], None, g[i], sg[i], 0, False, [x[i]], [], [acg[i]]))
            controller_buffer.add((x[i], None, acg[i], None, sg[i], None, 0, 0, [], [], []))

        return manager_buffer, controller_buffer