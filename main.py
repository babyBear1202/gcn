import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import random
from stgcn import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
use_gpu = True
num_timesteps_input = 50
num_timesteps_output = 10

# epochs = 1000
epochs=1000
batch_size = 1000

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
# if args.enable_cuda and torch.cuda.is_available():
#     args.device = torch.device('cuda')
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print(args.enable_cuda)
print(torch.cuda.is_available())

def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    epoch_loss_sum = 0
    epoch_loss_num= 0
    net.train()
    # print("-----------------------------0---------------------------------")
    for i in range(0, training_input.shape[0], batch_size):
    # for i in range(10):

        optimizer.zero_grad()
        # print("-----------------------------1---------------------------------")
        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        current_X_batch = X_batch.to(device=args.device)
        current_y_batch = y_batch.to(device=args.device)
        # print("-----------------------------2---------------------------------")
        out = net(A_wave, current_X_batch)
        loss = loss_criterion(out, current_y_batch)
        if i%1000 ==0:
            print(f'{i}/{training_input.shape[0]}: step loss:{loss.item()}')
        # print("-----------------------------3---------------------------------")
        loss.backward()
        # print("-----------------------------4---------------------------------")
        optimizer.step()
        # print("-----------------------------5---------------------------------")
        # epoch_training_losses.append(loss.clone().detach().cpu().numpy())
        epoch_loss_sum+=loss.item()
        epoch_loss_num+=1
        # print("-----------------------------6---------------------------------")
    # return sum(epoch_training_losses)/len(epoch_training_losses)
    return epoch_loss_sum/epoch_loss_num


if __name__ == '__main__':
    torch.manual_seed(7)
    print(args.device)
    A, X, means, stds = load_metr_la_data()
    print(A.shape)
    split_line1 = int(len(X) * 0.6)
    split_line2 = int(len(X) * 0.8)
    #
    train_original_data = X[:split_line1]
    train_mean, train_std = means[:split_line1], stds[:split_line1]

    val_original_data = X[split_line1:split_line2]
    val_mean, val_std = means[split_line1:split_line2], stds[split_line1:split_line2]

    test_original_data = X[split_line2:]
    test_mean, test_std = means[split_line2:], stds[split_line2:]

    training_input, training_target, train_mean_t, train_std_t = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output,
                                                       means=train_mean,
                                                       stds=train_std)
    # val_input, val_target, val_mean_t, val_std_t = generate_dataset(val_original_data,
    #                                          num_timesteps_input=num_timesteps_input,
    #                                          num_timesteps_output=num_timesteps_output,
    #                                          means=val_mean,
    #                                          stds=val_std)
    # test_input, test_target, test_mean_t, test_std_t = generate_dataset(test_original_data,
    #                                            num_timesteps_input=num_timesteps_input,
    #                                            num_timesteps_output=num_timesteps_output,
    #                                            means=test_mean,
    #                                            stds=test_std)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    # A_wave = torch.memory_format(padding)
    A_wave = A_wave.to(device=args.device)

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        print(f'{epoch}: epoch loss:{loss}')
        training_losses.append(loss)
        time.sleep(0.003)

    torch.save(net, "model/m_xb.pkl")
    print("Training loss: {}".format(np.stack(training_losses).mean()))
    plt.plot(training_losses, label="training loss")
    plt.legend()
    plt.show()
    with open("losses_20.pk", "wb") as fd:
        fd.write(training_losses)