from collections import deque, namedtuple
import math
from pprint import pp
import sys
import threading
import time
import cv2
import numpy as np
import pygame
from tetriZzz_Otimizado import TetriZzz_Otimizado
import random as rd
import os
import torch.nn as nn
import torch 
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):
    def __init__(self, capacidade):
      self.memory_neg = deque([], maxlen=int(capacidade/2))
      self.memory_pos = deque([], maxlen=int(capacidade/2))
      self.capacidade = capacidade

    def push(self, *args):
        reward = args[3]
        if reward > 0:
            if len(self.memory_pos) < self.capacidade/2:
                self.memory_pos.append(Transition(*args))
        else:
            if len(self.memory_neg) < self.capacidade/2:
                self.memory_neg.append(Transition(*args))

    def sample(self, batch_size):
    #   sampl = rd.sample(self.memory, batch_size)
    #   self.memory = deque([transition for transition in self.memory if transition.reward >= 6])

    #   return sampl
        sampl = deque(maxlen=int(self.capacidade))
        sampl.extend(self.memory_neg)
        sampl.extend(self.memory_pos)

        for _ in range(int(batch_size/2)):
            self.memory_neg.pop()
            self.memory_pos.pop()

        return sampl
    
    def __len__(self):
      return len(self.memory_neg) + len(self.memory_pos)

class ModeloBot(nn.Module):
    def __init__(self):
        super(ModeloBot, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=6, stride=1, padding=1)
        self.conv1_output = None
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=1)
        self.conv2_output = None
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=1)
        self.conv3_output = None

        self.fc1 = nn.Linear(64 * 18 * 11, 12800)
        self.fc2  = nn.Linear(12800, 1400)
        self.fc3  = nn.Linear(1400, 512)
        self.fc4  = nn.Linear(512, 512)
        self.fc5  = nn.Linear(512, saida)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x * torch.sigmoid(x)
        self.conv1_output = x

        x = self.conv2(x)
        x = x * torch.sigmoid(x)
        self.conv2_output = x

        x = self.conv3(x)
        x = x * torch.sigmoid(x)
        self.conv3_output = x

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

num_filters_c1 = 32
num_filters_c2 = 64
num_filters_c3 = 64
ncols = 16
nrows_c1 = (num_filters_c1 + ncols - 1) // ncols
nrows_c2 = (num_filters_c2 + ncols - 1) // ncols
nrows_c3 = (num_filters_c3 + ncols - 1) // ncols
nrows, num_filters = nrows_c1, num_filters_c1

def plotar_features(layer):
    if layer == None:
        return
    layer_np = layer[0].detach().cpu().numpy()
    # layer_np = np.transpose(layer_np, (3, 24, 17))
    # layer_np = np.squeeze(layer_np)

    filter_height, filter_width = layer_np[0].shape
    canvas_height = filter_height * nrows
    canvas_width = filter_width * ncols
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)  # Cria uma imagem preta

    # Preenche a imagem de canvas com os filtros
    for i in range(min(num_filters, ncols * nrows)):  # Limita a visualização ao número de filtros
        row = i // ncols  # Determina a linha do filtro na grade
        col = i % ncols   # Determina a coluna do filtro na grade
        filter_img = layer_np[i]  # Pega o filtro i
        filter_img = cv2.normalize(filter_img, None, 0, 255, cv2.NORM_MINMAX)  # Normaliza para a faixa 0-255
        filter_img = np.uint8(filter_img)  # Converte para inteiro de 8 bits

        # Coloca o filtro na posição correta na imagem de canvas
        canvas[row * filter_height : (row + 1) * filter_height, 
               col * filter_width : (col + 1) * filter_width] = filter_img

    # Exibe a imagem concatenada com os filtros
    cv2.imshow(f"Features", canvas)
    cv2.waitKey(10)

def criar_janela():
  os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (10, 10)
  pygame.init()
  info = pygame.display.Info()
  pygame.display.set_caption("TetriZzz")
#   return pygame.display.set_mode([info.current_w-300, info.current_h-500])
  return pygame.display.set_mode([300, 500])

BATCH_SIZE = 30
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000
TAU = 0.005
LR = 0.001

device = "cuda"

janela = criar_janela()
saida = 5
total_de_bots = 1000
total_escolhidos = 1
ativos = total_escolhidos

# melhores_n = [[] for _ in range(total_escolhidos)]
# plt.ion()
# fig, ax = plt.subplots()
# lines = [ax.plot([], [])[0] for _ in melhores_n]

mutacao = 0.002
colunas = 1
linhas = 1
# if total_de_bots < colunas:
#   colunas = total_de_bots
# linhas = total_de_bots % colunas
# if linhas == 0: linhas = total_de_bots / colunas
# else: linhas = int(total_de_bots / colunas) + 1
info = pygame.display.Info()
largura_dos_jogos = (info.current_w)/colunas
altura_dos_jogos = (info.current_h)/linhas

env = TetriZzz_Otimizado(0*largura_dos_jogos, (0+1)*largura_dos_jogos, 0*altura_dos_jogos, (0+1)*altura_dos_jogos, janela, desenhar=True)

policy_net = ModeloBot().to(device)
target_net = ModeloBot().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(BATCH_SIZE)
steps_done = 0

def select_action(state):
    global steps_done
    sample = rd.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[rd.randint(0, 4)]], device=device, dtype=torch.long)

def optimize_model():
    lmem = len(memory)
    if lmem < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()

cv2.namedWindow('Features', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Features', 800, 800)
debug = True
camada1 = True
ftr = 1
plot = True

for i_episode in count():
    # Initialize the environment and get its state
    env = TetriZzz_Otimizado(0*largura_dos_jogos, (0+1)*largura_dos_jogos, 0*altura_dos_jogos, (0+1)*altura_dos_jogos, janela, desenhar=True)
    state = env.tabuleiro.visao
    

    state = torch.tensor(state, dtype=torch.float32, device=device)
    state = state.unsqueeze(0).unsqueeze(0)
    for t in count():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("Saindo...")
                    pygame.quit()
                    plt.ioff()
                    cv2.destroyAllWindows()
                    print('Complete')
                    sys.exit(0)
                elif event.key == pygame.K_d:
                    debug = not debug
                    print(f"Debug: {debug}")
                elif event.key == pygame.K_e:
                    plot = False
                    cv2.destroyAllWindows()
                elif event.key == pygame.K_w:
                    plot = True
                    cv2.namedWindow('Features', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Features', 800, 800)
                elif event.key == pygame.K_1:
                    nrows, num_filters = nrows_c1, num_filters_c1
                    ftr = 1
                    print(f"Features 1")
                elif event.key == pygame.K_2:
                    nrows, num_filters = nrows_c2, num_filters_c2
                    ftr = 2
                    print(f"Features 2")
                elif event.key == pygame.K_3:
                    nrows, num_filters = nrows_c3, num_filters_c3
                    ftr = 3
                    print(f"Features 3")

        if t % 8 == 0:
           action = torch.tensor([[2]], device=device, dtype=torch.long)
        else:
            action = select_action(state)

        observation, reward, terminated = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # Store the transition in memory
        rwrd_itm = reward.item()
        if rwrd_itm != 0:
            memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
        if loss is not None and debug:
            print(f"Epoch {i_episode}, Iteration {t}, Loss {loss}, Reward {rwrd_itm}")

        if debug and plot:
            if ftr == 1:
                plotar_features(policy_net.conv1_output)
            elif ftr == 2:
                plotar_features(policy_net.conv2_output)
            elif ftr == 3:
                plotar_features(policy_net.conv3_output)
        # if i_episode % 100 == 0:
        #     plotar_features(policy_net.conv2_output, "Conv2", t)


        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break
    print(f"{env.tabuleiro.pontos}")

