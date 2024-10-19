import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import RNN, TrackData
import torch.optim
import os
import numpy as np
import matplotlib.pyplot as plt

isDebug = False


def load_dataset(dirpath, state_len=50, pred_len=10, features=3):  # 后两个参数的意思是用state_len长度的序列推测pred_len长度的序列
    files = os.listdir(dirpath)
    Dataset = torch.zeros(1, state_len + pred_len, features)
    for file in files:
        with open(os.path.join(dirpath, file), 'r') as f:
            lines = f.readlines()
            linedata = []  # 一组数据，数据最基本的单位
            for line in lines:
                strList = line.split('\t')
                numList = [float(i) for i in strList]
                if len(numList) == 3:
                    linedata.append(numList)
                if len(linedata) == (state_len + pred_len):
                    blockArray = np.array(linedata)
                    blockTensor = torch.from_numpy(blockArray).float()
                    blockTensor = blockTensor.unsqueeze(0)
                    Dataset = torch.cat((Dataset, blockTensor), 0)
                    linedata.clear()
    print("Sample:{}".format(len(Dataset)))
    dataset = TrackData(Dataset)
    return dataset


def start_train(data_iter, val_iter, net, lossfunc, optimizer, epochs, device):
    lossArray = []  # 存储损失值，画图
    history_min = 100000000
    pred_len = 10
    state_len = 50
    net = net.to(device)
    net.train()
    lossfunc.to(device)
    for epoch in range(epochs):
        lossSum = 0
        count = 0
        for data in data_iter:
            data = data.to(device)
            x = data[:, 0:state_len, :]
            y = data[:, state_len:state_len + pred_len, :]
            y_hat = net(x)
            loss = lossfunc(y_hat, y)
            loss.backward()
            optimizer.step()
        for val in val_iter:
            val = val.to(device)
            x = val[:, 0:state_len, :]
            y = val[:, state_len:state_len + pred_len, :]
            y_hat = net(x)
            loss = lossfunc(y_hat, y)
            lossSum = lossSum + loss.item()
            count = count + 1
        print("Epoch[{}/{}]-----Loss[{}]".format(epoch, epochs, lossSum / count))
        lossArray.append(lossSum / count)
        if(lossSum / count < history_min):
            history_min = lossSum / count
            torch.save(net.state_dict(), 'runs/best.pth')
            print('---------------------------------Best Update---------------------------------')
    plt.plot(lossArray)
    plt.title("Loss vs Epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss-iter.png')
    plt.show()
    torch.save(net.state_dict(), 'runs/last.pth')


if __name__ == '__main__':
    state_len = 990
    pred_len = 10
    net = RNN(input_size=3, hidden_size=state_len, output_features=3, pred_len=pred_len, num_layers=1)
    #param = torch.load('runs/last.pth')
    #net.load_state_dict(param)
    loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = load_dataset('data/train',state_len=state_len, pred_len=pred_len)
    valid_dataset = load_dataset('data/val',state_len=state_len, pred_len=pred_len)
    data_iter = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_iter = DataLoader(valid_dataset, batch_size=1024, shuffle=False)
    start_train(data_iter, val_iter, net, loss, optimizer, epochs=500, device=device)
