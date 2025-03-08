import torch.nn.functional as F
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, input_shape)
        self.bn1 = nn.LayerNorm(input_shape)

        self.fc2 = nn.Linear(input_shape, 128)
        self.bn2 = nn.LayerNorm(128)

        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.LayerNorm(128)

        self.fc4 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        return self.fc4(x)