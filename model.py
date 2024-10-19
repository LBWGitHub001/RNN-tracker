import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, pred_len=10, output_features=3, num_layers=1):
        super(RNN, self).__init__()
        # N批量 L时间维 H特征维
        # 输入的形状是(N,L,H)
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.output_features = output_features
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_features * pred_len*2)
        self.fc2 = nn.Linear(output_features * pred_len*2, output_features * pred_len)

    def forward(self, x):
        means = x.mean()
        h0 = means*torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        # 前向传播GRU
        out, _ = self.gru(x, h0)
        out=out[:, -1, :]
        y_hidden = out.reshape(out.size(0), -1)
        y_hat = self.fc1(y_hidden)
        y_hat = F.relu(y_hat)
        y_hat = self.fc2(y_hat)
        y_hat = y_hat.view(y_hat.size(0), self.pred_len, self.output_features)
        return y_hat


class TrackData(Dataset):
    def __init__(self, data):
        super(TrackData, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :, :]
