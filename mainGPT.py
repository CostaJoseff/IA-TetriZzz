from collections.abc import Sequence
import torch, pygame, time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from gym.spaces import Discrete, Box
from gym import Env
# from Jogo.TetriZzz_Otimizado import TetriZzz_Otimizado
from Jogo2.TetriZzz import TetriZzz
from AI_Hub.valores import tamanho_fila_movimentos
import seaborn as sns
import matplotlib.pyplot as plt
from threading import Thread
plt.ion()
plt.figure(figsize=(2, 5))

class FilaDeMovimentos():
    def __init__(self, tamanho):
        self.fila = [-1]*tamanho
        self.tamanho = tamanho

    def em_fila(self, elemento):
        self.fila.append(elemento)
        self.fila.pop(0)

    def to_numpy(self):
        return np.array(self.fila)

class Memory_():
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.dequeue_pos = deque(maxlen=int(mem_size/2))
        self.dequeue_neg = deque(maxlen=int(mem_size-len(self.dequeue_pos)))

    def __len__(self):
        return len(self.dequeue_neg) + len(self.dequeue_pos)
    
    def len_(self):
        a = 0
        b = 0
        size = int(batch_size/2)
        size_complm = batch_size - size
        if len(self.dequeue_neg) >= size:
            a = size
        else:
            a = len(self.dequeue_neg)

        if len(self.dequeue_pos) >= size_complm:
            b = size_complm
        else:
            b = len(self.dequeue_pos)

        return a + b
    
    def __str__(self):
        return f"Pos - {len(self.dequeue_pos)} -- Neg - {len(self.dequeue_neg)}"
    
    def sample(self, batch_size):
        pos_sample = [self.dequeue_pos.popleft() for _ in range(min(len(self.dequeue_pos), batch_size // 2))]
        neg_sample = [self.dequeue_neg.popleft() for _ in range(min(len(self.dequeue_neg), batch_size - len(pos_sample)))]


        final_list = pos_sample+neg_sample
        random.shuffle(final_list)
        return final_list

    def append(self, data):
        if data[2] >= 0:
            self.dequeue_pos.append(data)
        else:
            self.dequeue_neg.append(data)
    
    def __iter__(self):
        return iter(list(self.dequeue_pos) + list(self.dequeue_neg))
    
import torch.nn.functional as F
import torch.nn as nn
from torch import flatten

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(49152, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)

        x = flatten(x, start_dim=1)
        return self.fc(x)

def update_plot(model, layer="fc.weight"):
    weights = model.state_dict()[layer].cpu().numpy()
    sns.heatmap(weights, annot=False, cmap="magma", fmt=".4f", cbar=False)
    plt.title(f"Distribuição de Pesos - {layer}")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.pause(0.1)

def criar_janela():
    pygame.init()
    info = pygame.display.Info()
    pygame.display.set_caption("TetriZzz")
    return pygame.display.set_mode([info.current_w/4, info.current_h/2])

# Parâmetros
env = TetriZzz(True)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
learning_rate = 5e-4
batch_size = 20
memory_size = 10000
memory = Memory_(memory_size)

global quit
quit = False

# Modelo e otimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(state_size, action_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Função de treino
def train(ep):
    if memory.len_() < batch_size:
        return batch_size, ep

    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = model(next_states).max(1)[0].detach()
    targets = rewards + gamma * next_q_values * (1 - dones)

    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    update_plot(model)

    n_bsz = int(batch_size * 1.5)
    return  min(n_bsz, 500), ep + 1

# Treinamento do agente
ep = 0
maior = -float("inf")
maior_score = 0
print(f"Iniciando treinamento")
user_input = True
flag = True
while True:
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if user_input:
            flag = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    flag = True
                    if event.key == pygame.K_w:
                        action = 0
                    elif event.key == pygame.K_a:
                        action = 1
                    elif event.key == pygame.K_s:
                        action = 2
                    elif event.key == pygame.K_d:
                        action = 3
                    elif event.key == pygame.K_SPACE:
                        action = 4
                    elif event.key == pygame.K_q:
                        action = 2
                        user_input = False
                        flag = True
                        continue
                    else:
                        continue
        else:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()

        if not flag:
            continue

        next_state, reward, done, _ = env.step(action)
        print(reward)
        # env.jogo.tabuleiro.pontos += reward
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        # batch_size, ep = train(ep)
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay) if ep % 10 == 0 else epsilon
    maior = total_reward if total_reward > maior else maior
    # maior_score = env.jogo.tabuleiro.pontos if env.jogo.tabuleiro.pontos > maior_score else maior_score

    

# Teste do agente treinado
for _ in range(10):
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = torch.argmax(model(state_tensor)).item()
        state, _, done, _ = env.step(action)
