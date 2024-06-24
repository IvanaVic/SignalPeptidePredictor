import os.path

import torch
import torch.nn as nn


class SignalPeptidePredictorCNN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SignalPeptidePredictorCNN2, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4*hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim[0] * output_dim[1])
        self.fc_org = nn.Linear(3, hidden_dim)
        self.fc_seq = nn.Linear(hidden_dim*4*73, hidden_dim*3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, y):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc_seq(x))
        y = self.relu(self.fc_org(y))
        xy = torch.concat([x, y], dim=-1)
        xy = self.relu(self.fc1(xy))
        xy = self.dropout(xy)
        out = self.fc2(xy)
        return out
