import os
import sys
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import copy
import pandas as pd
import random
import torch.utils.data as Data
from twin_stgcn import STGCN
# from stgcn import STGCN
from utils import DianDataset,get_normalized_adj,load_metr_la_data
import logging

# logging config
LOG_FILE = 'train.log'
logger = logging.getLogger(__file__)
FORMAT = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
format_str = logging.Formatter(FORMAT)  # 设置日志格式
logger.setLevel(logging.DEBUG)  # 设置日志级别
sh = logging.StreamHandler(stream=sys.stdout)  # 往屏幕上输出
fh = logging.FileHandler(filename=f'LOG/{LOG_FILE}', mode='w', encoding='utf-8')
sh.setFormatter(format_str)  # 设置屏幕上显示的格式
fh.setFormatter(format_str)
logger.addHandler(sh)  # 把对象加到logger里
logger.addHandler(fh)

# set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# training config
use_gpu = True
num_timesteps_input = 50
num_timesteps_output = 10
epochs = 1518
batch_size = 128
feature_num = 5
current_epoch = 0

# handle input argument
parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
    # logger.info('Using GPU')
else:
    args.device = torch.device('cpu')
    # logger.info('Using CPU')


def train_epoch(optimizer, net, training_input, training_target):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    # permutation = torch.randperm(training_input.shape[0])

    # epoch_training_losses = []
    # net.train()
    # logger.info("-----------------------------0---------------------------------")
    # for i in range(0, batch_size):
    # for i in range(10):

    # optimizer.zero_grad()
    # logger.info("-----------------------------1---------------------------------")
    # indices = permutation[i:i + batch_size]
    current_X_batch, current_y_batch = training_input, training_target
    # logger.info("-----------------------------2---------------------------------")
    out = net(A_wave,ele_A_wave, current_X_batch)
    loss = loss_criterion(out, current_y_batch)
        # logger.info(f'{i}/{training_input.shape[0]}: step loss:{loss.item()}')
        # print(f'{i}/{training_input.shape[0]}: step loss:{loss.item()}')
    # logger.info(f'2--------------------out:{out},current_y_batch:{current_y_batch}')
    loss.backward()
    # logger.info(f'3--------------------loss:{loss}')
    # logger.info("-----------------------------4---------------------------------")
    optimizer.step()
    # logger.info(f'4--------------------loss:{loss}')
    # logger.info("-----------------------------5---------------------------------")
    # epoch_training_losses.append(loss.clone().detach().cpu().numpy())
    #     logger.info("-----------------------------6---------------------------------")
    # return sum(epoch_training_losses)/len(epoch_training_losses)
    return loss


def split_train_val_test(X, means, stds):
    split_line1 = int(len(X) * 0.6)
    split_line2 = int(len(X) * 0.8)
    #
    train_original_data = X[:split_line1]
    train_mean, train_std = means[:split_line1], stds[:split_line1]

    val_original_data = X[split_line1:split_line2]
    val_mean, val_std = means[split_line1:split_line2], stds[split_line1:split_line2]

    test_original_data = X[split_line2:]
    test_mean, test_std = means[split_line2:], stds[split_line2:]
    return (train_original_data, train_mean, train_std), \
           (val_original_data, val_mean, val_std ), \
           (test_original_data, test_mean, test_std)

def cal_mean_std():
    result = np.zeros([1035,39,5,1001])
    files = os.listdir('np_dir')
    for i,file in enumerate(files):
        X = np.load('np_dir/'+file).transpose((0, 2, 1)).astype(np.float32)
        result[i] = X
        # result_sum = result_sum + X
    # result_sum = result_sum/len(files)
    mean = np.mean(result, axis=(0, 1, 3))
    std = np.std(result, axis=(0, 1, 3))
    return mean,std


if __name__ == '__main__':
    logger.info("start loading dataset...")
    means, stds = cal_mean_std()
    print(means,stds)
    dataset = DianDataset(logger,'C:/From D/电网/电网代码/STGCN-PyTorch-dianwang/np_dataset','train',num_timesteps_input,num_timesteps_output,means,stds)
    val_dataset = DianDataset(logger,'C:/From D/电网/电网代码/STGCN-PyTorch-dianwang/np_dataset','val',num_timesteps_input,num_timesteps_output,means,stds)

    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=32,
        pin_memory=True)
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=32,
        pin_memory=True)
    logger.info(f"finish loading dataset, dataset length:{len(dataset)}")

    #############################ele distance matrix###################
    ele_A = np.load('Ybus_inv.npy')
    for i in range(0,ele_A.shape[0]):
        for j in range(0,ele_A.shape[1]):
            ele_A[i,j] = ele_A[i,i]-2*ele_A[i,j]-ele_A[j,j]
    # print(ele_A)
    ele_A_wave = get_normalized_adj(ele_A)
    ele_A_wave = torch.from_numpy(ele_A_wave)
    ele_A_wave = ele_A_wave.to(device=args.device)
    ###################################################################

    #############################adjacency matrix###################
    A = pd.read_excel('AdjacencyMatrix.xls').values.astype(np.float32)
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=args.device)
    ###################################################################

    print("starting a new model")
    net = STGCN(A_wave.shape[0],
            feature_num,
            num_timesteps_input,
            num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)
    optimizer.zero_grad()

    loss_criterion = nn.MSELoss()

    training_losses = []
    val_losses = []
    epoch_training_losses = []
    validation_losses = []
    validation_maes = []
    for epoch in range(current_epoch,epochs):
        net.train()
        print(f"current epoch ----------------{epoch}-----------------------------")
        for batch_dix, sample in enumerate(loader):
            training_input =torch.from_numpy(np.array(sample["train_input"])).to(device=args.device)
            training_target = torch.from_numpy(np.array(sample["train_output"])).to(device=args.device)
            loss = train_epoch(optimizer, net, training_input, training_target)
            # logger.info(f'idx: {batch_dix},epoch: {epoch}  loss:{loss}')
            training_losses.append(loss.clone().detach().cpu().numpy())
        epoch_training_loss = np.sum(training_losses)/len(training_losses)
        logger.info(f'{epoch}: epoch loss:{epoch_training_loss}')
        # print(f'{epoch}: epoch loss:{epoch_training_loss}')
        epoch_training_losses.append(epoch_training_loss)

        #######################validation#########################################o
        with torch.no_grad():
            net.eval()
            for batch_dix, sample in enumerate(val_loader):
                val_input =torch.from_numpy(np.array(sample["train_input"])).to(device=args.device)
                val_target = torch.from_numpy(np.array(sample["train_output"])).to(device=args.device)
                out = net(A_wave, ele_A_wave, val_input)
                unnorma_val_target = (val_target * stds[0] + means[0]).clone().detach().cpu().numpy()
                unnorma_out = (out * stds[0]+means[0]).clone().detach().cpu().numpy()
                loss = loss_criterion(out, val_target)
                mae = np.mean(np.absolute(unnorma_val_target - unnorma_out))
                np.savez(f'test/{batch_dix}.npy', val_input=val_input.clone().detach().cpu().numpy(),
                         val_target=val_target.clone().detach().cpu().numpy(),
                         out=out.clone().detach().cpu().numpy(),
                         unnorma_out=unnorma_out,
                         unnorma_val_target = unnorma_val_target)
                # logger.info(f'out is :{out[0]}, val_target is :{val_target[0]}')
                # logger.info(f'idx: {batch_dix},epoch: {epoch}  loss:{loss}')
                val_losses.append(loss.clone().detach().cpu().numpy())
                validation_maes.append(mae)
            epoch_val_loss = np.sum(val_losses) / len(val_losses)
            epoch_mae_loss = np.sum(validation_maes) / len(validation_maes)
            logger.info(f'{epoch}: epoch val loss:{epoch_val_loss}')
            logger.info(f'{epoch}: epoch mae loss:{epoch_mae_loss}')
        ##########################################################################
        checkpoint_path = "checkpoint_combine_distance"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        file_path = os.path.join('checkpoint_combine_distance', str(epoch))
        with open(file_path, "wb") as fd:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_training_loss
                },fd)

    torch.save(net, "model/m_combine_attention.pkl")

    if not os.path.exists("losses"):
        os.makedirs("losses")
    with open("losses/combine_loss", "wb") as fd:
        fd.write(epoch_training_losses)

    logger.info("Training loss: {}".format(np.stack(epoch_training_losses).mean()))
    plt.plot(epoch_training_losses, label="training loss")
    plt.legend()
    plt.show()