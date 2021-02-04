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
num_timesteps_input = 20
num_timesteps_output = 5

epochs = 1000
# epochs=1
batch_size = 16

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
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        print(f'{i}/{training_input.shape[0]}: step loss:{loss.item()}')
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(7)
    print(args.device)
    A, X, means, stds = load_metr_la_data()
    print(A.shape)
    split_line1 = int(len(X) * 0.6)
    split_line2 = int(len(X) * 0.8)
    #
    train_original_data = X[:split_line1]

    val_original_data = X[split_line1:split_line2]
    # val_mean, val_std = means[split_line1:split_line2], stds[split_line1:split_line2]

    test_original_data = X[split_line2:]
    # test_mean, test_std = means[split_line2:], stds[split_line2:]

    training_input, training_target, train_mean_t, train_std_t = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output,
                                                       means=means,
                                                       stds=stds)
    val_input, val_target, val_mean_t, val_std_t = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output,
                                             means=means,
                                             stds=stds)
    test_input, test_target, test_mean_t, test_std_t = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output,
                                               means=means,
                                               stds=stds)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    # A_wave = torch.memory_format(padding)
    A_wave = A_wave.to(device=args.device)

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            torch.save(net, "model/m_2_globalmean.pkl")
            print("Training loss: {}".format(training_losses[-1]))
        #     net.eval()
        #     val_input = val_input.to(device=args.device)
        #     val_target = val_target.to(device=args.device)
        #
        #     torch.cuda.empty_cache()
        #     out = net(A_wave, val_input)
        #     torch.cuda.empty_cache()
        #     val_loss = loss_criterion(out, val_target).to(device="cuda")
        #     validation_losses.append(np.asscalar(val_loss.detach().numpy()))
        #
        #     out_unnormalized = out.detach().cpu().numpy()*val_std_t+val_mean_t
        #     target_unnormalized = val_target.detach().cpu().numpy()*val_std_t+val_mean_t
        #     mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        #     validation_maes.append(mae)
        #
        #     out = None
        #     val_input = val_input.to(device="cuda")
        #     val_target = val_target.to(device="cuda")
        #
        # print("Training loss: {}".format(training_losses[-1]))
        # print("Validation loss: {}".format(validation_losses[-1]))
        # print("Validation MAE: {}".format(validation_maes[-1]))
        # plt.plot(training_losses, label="training loss")
        # plt.plot(validation_losses, label="validation loss")
        # plt.legend()
        # plt.show()
        #
        # checkpoint_path = "checkpoints/"
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path)
        # with open("checkpoints/losses_20.pk", "wb") as fd:
        #     pk.dump((training_losses, validation_losses, validation_maes), fd)
    # torch.save(net,"model/m_1.pkl")