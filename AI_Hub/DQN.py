from threading import Semaphore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten

def init_linear_layer(m, method='tanh'):
    if isinstance(m, nn.Linear):
        if method == "tanh":
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(method))
            nn.init.constant_(m.bias, 0)
        elif method == "sm_normal":
            nn.init.uniform_(m.weight, a=0.0001, b=0.001)
            nn.init.constant_(m.bias, 0.0)

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions, input_shape):
        super(DQN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_actions = num_actions
        self.mutex = Semaphore(1)

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.mpool = nn.MaxPool2d(kernel_size=2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *input_shape)
            dummy_output = self._forward_conv(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.flattened_size, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1)
            ) for _ in range(num_actions)
        ])

    def _forward_conv(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = self.mpool(x)
        return x

    def forward(self, x):
        self.mutex.acquire()
        x = self._forward_conv(x)
        x = flatten(x, start_dim=1)

        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=1)
        self.mutex.release()
        return out

# class DQN(nn.Module):
#     def __init__(self, input_channels, num_actions, input_shape):
#         super(DQN, self).__init__()

#         self.mutex = Semaphore(1)

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=6, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)

#         self.mpool = nn.MaxPool2d(kernel_size=2)
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, input_channels, *input_shape)
#             x = self._forward_conv(dummy_input)
#             flattened_size = x.view(1, -1).shape[1]

#         self.fc1 = nn.Linear(flattened_size, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 32)
#         self.fc4 = nn.Linear(32, num_actions)

#         self.apply(lambda m: init_linear_layer(m, method='sm_normal')) # leaky_relu

#     def _forward_conv(self, x):
#         x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
#         x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
#         x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
#         x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
#         x = self.mpool(x)
#         return x

#     def forward(self, x):
#         self.mutex.acquire()
#         x = self._forward_conv(x)
#         x = flatten(x, start_dim=1)
#         x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
#         x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
#         x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
#         x = self.fc4(x)
#         self.mutex.release()
#         return x
