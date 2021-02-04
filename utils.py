import os
import zipfile

import numpy as np
np.set_printoptions(suppress=True)


import torch
import pandas as pd


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
