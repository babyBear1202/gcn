import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn

from stgcn import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj
use_gpu = True


parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
# if args.enable_cuda and torch.cuda.is_available():
#     args.device = torch.device('cuda')
# if torch.cuda.is_available():
#     args.device = torch.device('cuda')
# else:
args.device = torch.device('cuda')
print(args.enable_cuda)
print(torch.cuda.is_available())
num_timesteps_input = 50
num_timesteps_output = 10
validation_losses = []
validation_maes = []
test_losses = []
test_maes = []
loss_criterion = nn.MSELoss()
if __name__ == '__main__':
        net = torch.load('model/m_xb.pkl')
        net.eval()
        net.to(args.device)
        # print("training loss:{}".format())
        A, X, means, stds = load_metr_la_data()

        A_wave = get_normalized_adj(A)
        A_wave = torch.from_numpy(A_wave)
        # A_wave = torch.memory_format(padding)
        A_wave = A_wave.to(device=args.device)

        split_line1 = int(len(X) * 0.6)
        split_line2 = int(len(X) * 0.8)



        val_original_data = X[split_line1:split_line2]
        val_mean, val_std = means[split_line1:split_line2], stds[split_line1:split_line2]

        val_input, val_target, val_mean_t, val_std_t = generate_dataset(val_original_data,
                                                                        num_timesteps_input=num_timesteps_input,
                                                                        num_timesteps_output=num_timesteps_output,
                                                                        means=val_mean,
                                                                       stds=val_std)
        test_original_data = X[split_line2:]
        test_mean, test_std = means[split_line2:], stds[split_line2:]

        test_input, test_target, test_mean_t, test_std_t = generate_dataset(test_original_data,
                                                                            num_timesteps_input=num_timesteps_input,
                                                                            num_timesteps_output=num_timesteps_output,
                                                                            means=test_mean,
                                                                            stds=test_std)

        batch_size = 100
        num_batches = val_target.shape[0] // batch_size
        for batch_idx in range(0, num_batches+1):
                current_input = val_input[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device=args.device)
                current_target = val_target[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device=args.device)
                current_mean, current_std = val_mean_t[batch_idx*batch_size:(batch_idx+1)*batch_size], val_std_t[batch_idx*batch_size:(batch_idx+1)*batch_size]
                # print(current_input.shape)
                # print(current_target.shape)
                current_out = net(A_wave, current_input)

                val_loss = loss_criterion(current_out, current_target).to(device=args.device)
                validation_losses.append(np.asscalar(val_loss.detach().cpu().numpy()))

                current_mean, current_std = current_mean.cpu().numpy(), current_std.cpu().numpy()
                current_out, current_target = current_out.detach().cpu().numpy(), current_target.cpu().numpy()
                print(current_out.shape)
                # print(mean.shape)
                # print(std.shape)
                # print(target_unnormalized.shape)
                out_unnormalized =  current_out * current_std[:, 0, None, None] + current_mean[:, 0,None, None]
                target_unnormalized = current_target*current_std[:,0,None,None]+current_mean[:,0,None,None]
                # print(out_unnormalized.shape)
                # print(target_unnormalized.shape)
                mae = np.absolute(out_unnormalized[:,29:,] - target_unnormalized[:,29:,])
                validation_maes.append(mae)

        print("Validation loss: {}".format(np.stack(validation_losses).mean()))
        print("Validation MAE: {}".format(np.concatenate(validation_maes,0).mean()))


        # //test
        batch_size = 50
        num_batches = test_target.shape[0] // batch_size
        for batch_idx in range(0, num_batches + 1):
                current_input =test_input[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device=args.device)
                current_target = test_target[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device=args.device)
                current_mean, current_std = test_mean_t[batch_idx * batch_size:(batch_idx + 1) * batch_size], test_std_t[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                # print(current_input.shape)
                # print(current_target.shape)
                current_out = net(A_wave, current_input)
                test_loss = loss_criterion(current_out, current_target).to(device=args.device)
                test_losses.append(np.asscalar(test_loss.detach().cpu().numpy()))

                mean, std = current_mean.unsqueeze(1).cpu().numpy(), current_std.unsqueeze(1).cpu().numpy()
                current_out, current_target = current_out.detach().cpu().numpy(), current_target.cpu().numpy()

                out_unnormalized = current_out * std + mean  # TODO: 这里不应该知道
                target_unnormalized = current_target * std + mean
                mae = np.absolute(out_unnormalized[:, :, 0] - target_unnormalized[:, :, 0])
                test_maes.append(mae)

        print("test loss: {}".format(np.stack(test_losses).mean()))
        print("test MAE: {}".format(np.concatenate(test_maes, 0).mean()))