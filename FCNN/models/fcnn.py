import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU 激活函数
        x = self.fc2(x)          # 输出层
        return x
