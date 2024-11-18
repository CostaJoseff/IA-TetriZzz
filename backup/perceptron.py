import torch
import tensorflow as tf

class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def train_model(self, model, input_data, reward):
        criterion = nn.SmoothL1Loss()
        optimizer = optim.SGD(model.parameters(), lr=1)

        optimizer.zero_grad()
        output = model(input_data)
        reward_tensor = torch.tensor([reward], dtype=torch.float)
        loss = criterion(output, reward_tensor)
        loss.backward()
        optimizer.step()

        return output, loss.item()