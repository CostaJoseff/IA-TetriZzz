import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import flatten

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions, input_shape):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)

        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(24576, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        # x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        # x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)

        x = flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)