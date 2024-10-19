import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import TrackData, RNN
import numpy as np
import argparse


def load_dataset(file_path, state_len=50, pred_len=10, features=3):  # 后两个参数的意思是用state_len长度的序列推测pred_len长度的序列
    with open(file_path, 'r') as f:
        lines = f.readlines()
        linedata = []  # 一组数据，数据最基本的单位
        for line in lines:
            strList = line.split('\t')
            numList = [float(i) for i in strList]
            if len(numList) == 3:
                linedata.append(numList)
    blockArray = np.array(linedata)
    Dataset = torch.from_numpy(blockArray).float()
    print("Sample:{}".format(len(Dataset)))
    return Dataset


def get_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--input-size", type=int, default=3, help="input size")
    parser.add_argument("--hidden-size", type=int, default=900, help="hidden size")
    parser.add_argument("--output-features", type=int, default=3, help="output size")
    parser.add_argument("--pred-len", type=int, default=10, help="predict length")
    parser.add_argument("--nums-layer", type=int, default=2, help="nums of layers")
    parser.add_argument("--data", type=str, default='data/val/3.txt')
    parser.add_argument("--model", type=str, default='runs/best.pth')
    args = parser.parse_args()
    return args


def export(args):
    # 检查cuda是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset(args.data)
    net = RNN(input_size=args.input_size, hidden_size=args.hidden_size, output_features=args.output_features,
              pred_len=args.pred_len,num_layers=args.nums_layer)
    param = torch.load(args.model)
    net.load_state_dict(param)
    net.eval()
    net = net.to(device)

    # 计算
    data = dataset
    data = data.to(device)
    data = data[-args.hidden_size - 1:-1, :]
    data = data.unsqueeze(0)
    y_hat = net(data)
    y_hat = y_hat.reshape(args.pred_len, args.output_features)
    y_hat = y_hat.to('cpu').detach().numpy()
    # y = torch.cat(dataset, y_hat).to('cpu')
    plt.plot([i for i in range(len(dataset[:, 0]))], dataset[:, 0], color = 'b')
    plt.plot([len(dataset[:, 0])+i+1 for i in range(len(y_hat[:, 0]))], y_hat[:, 0], color = 'r')
    plt.title("X direction predect")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('X.png')
    plt.show()


if __name__ == '__main__':
    args = get_args()
    export(args)
