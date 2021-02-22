import os
import zipfile

import numpy as np
np.set_printoptions(suppress=True)


import torch
import pandas as pd

from torch.utils.data.dataset import Dataset
from typing import Union, Optional
from pathlib import Path
import glob
import os
import numpy as np
import logging

from typing import TypeVar



class DianDataset(torch.utils.data.Dataset):

    def __init__(self, logger: Optional[logging.Logger], dataset_dir: Union[str, Path], dataset_type: str,
                 num_input_timestep:int, num_predict_timestep:int) -> None:
        super().__init__()
        self.logger = logger
        assert dataset_type in ['train', 'val', 'test']
        self.npy_data_list = np.array(sorted([os.path.join(dataset_dir, x) for x in glob.glob(dataset_dir+'/*npy')]))
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.num_input_timestep = num_input_timestep
        self.num_predict_timestep = num_predict_timestep
        # split train/val/test
        num_npys = len(self.npy_data_list)
        # shuffle train/val/test
        npz_idxes = np.arange(num_npys)
        np.random.shuffle(npz_idxes)
        train_split, val_split, test_split = npz_idxes[:int(num_npys*0.6)], npz_idxes[int(num_npys*0.6):int(num_npys*0.8)], \
                                                npz_idxes[int(num_npys * 0.8):]
        self.dataset = {'train': self.npy_data_list[train_split],
                        'val': self.npy_data_list[val_split],
                        'test': self.npy_data_list[test_split]}[self.dataset_type]

        self.buffer = [None]*len(self.dataset)
        self.num_npys = len(self.dataset)
        self.num_samples_each_npy = self.cal_total_samples(self.get_npy(0)[0])

    def get_npy(self, np_index):
        if self.buffer[np_index] is not None:
            x = self.buffer[np_index]
        else:
            npy_path = self.dataset[np_index]
            x = np.load(npy_path).transpose((0, 2, 1)).astype(np.float32)
            self.buffer[np_index] = x
        mean = np.mean(x, axis=(0, 2))
        std = np.std(x, axis=(0, 2))
        import copy
        return copy.deepcopy(x), mean, std

    def generate_train_sample(self,X, sample_idx):
        train_input = X[:, sample_idx:sample_idx+self.num_input_timestep,:]
        train_output = X[:, sample_idx+self.num_input_timestep:sample_idx+self.num_input_timestep+self.num_predict_timestep,0]
        return torch.from_numpy(train_input), torch.from_numpy(train_output)

    def cal_total_samples(self, x) -> int:
         return x.shape[1] - self.num_input_timestep-self.num_predict_timestep+1

    def __len__(self):
        return self.num_npys * self.num_samples_each_npy

    def __getitem__(self, index):

        if self.logger:
            self.logger.debug(f'current sample:{index}')
        np_index, sample_index = index//self.num_samples_each_npy, index%self.num_samples_each_npy
        x, mean, std = self.get_npy(np_index)
        train_input, train_output = self.generate_train_sample(x, sample_index)
        sample = {
            'train_input': train_input,
            'train_output': train_output,
            'mean': mean,
            'std': std}
        # if index == 1916: print(sample)
        return sample



def load_metr_la_data():
    # if (not os.path.isfile("data/adj_mat.npy")
    #         or not os.path.isfile("data/node_values.npy")):
    #     with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
    #         zip_ref.extractall("data/")
    A = pd.read_excel('AdjacencyMatrix.xls').values.astype(np.float32)
    X = []
    means = []
    stds = []
    for file in os.listdir('np_dir'):
        file_path = os.path.join('np_dir', file)
        # print(file_path)
        tmp_x = np.load(file_path,allow_pickle=True).transpose((0, 2, 1)).astype(np.float32)
        ##deal with the load_node
        for i in range(0,29):
            fz = tmp_x[i][0][:]
            xj = tmp_x[i][1][:]
            tmp_x[i][1][:] = (tmp_x[i][1][:]%360) * np.pi/180
            tmp_x[i][2][:] = fz * np.cos(xj * np.pi / 180)

            tmp_x[i][3][:] = fz * np.sin(xj * np.pi / 180)
            cz_sb =tmp_x[i][2][1:] -  tmp_x[i][2][:-1]
            cz_xb =tmp_x[i][3][1:] -  tmp_x[i][3][:-1]
            # print(tmp_x[0, :, :5])

            cz_sb1 = cz_xb[1:] - cz_xb[:-1]
            cz_sb1[cz_sb1==0] = 1e-10
            cz_sb[cz_sb==0] = 1e-10

            diff = cz_xb / cz_sb
            diff[diff==0] = 1e-10
            s = diff[1:] - diff[:1]
            diff1 = s/cz_sb1
            diff1[diff1==0] = 1e-10
            one = np.ones(diff1.shape)
            p = np.fabs((one+diff[1:]**2)**(3/2)/diff1)
            tmp_x[i][4][2:] = p
        for j in range(29,39):
            tmp_x[j][0][:] = (tmp_x[j][0][:]%360) * np.pi/180

        tmp_means = np.mean(tmp_x, axis=(0, 2))
        tmp_x = tmp_x - tmp_means.reshape(1, -1, 1)
        tmp_stds = np.std(tmp_x, axis=(0, 2))
        tmp_x = tmp_x / tmp_stds.reshape(1, -1, 1)
        X.append(tmp_x)
        means.append(tmp_means)
        stds.append(tmp_stds)

    # Normalization using Z-score method
    # means = np.mean(X, axis=(0, 2))
    # X = X - means.reshape(1, -1, 1)
    # stds = np.std(X, axis=(0, 2))
    # X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points


    # Save samples
    features, target, fmeans, fstds = [], [], [], []
    for idx,x in enumerate(X):
        indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
                   in range(x.shape[2] - (
                    num_timesteps_input + num_timesteps_output) + 1)]
        for i, j in indices:
            features.append(
                x[:, :, i: i + num_timesteps_input].transpose(
                    (0, 2, 1)))
            target.append(x[:, 0, i + num_timesteps_input: j])
            fmeans.append(means[idx])
            fstds.append(stds[idx])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target)), \
           torch.from_numpy(np.array(fmeans)),\
           torch.from_numpy(np.array(fstds))
