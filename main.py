import os
from matplotlib import  pyplot as plt
plt.ion()
plt.figure(figsize=(10, 2))

from AI_Hub.valores import device

import torch.nn.functional as F
import torch.optim as O

from AI_Hub.DQN import DQN
from AI_Hub.TatrizEnv import TetrizEnv
from AI_Hub.AI_Hub import AI_Hub

modelos = []
ambientes = []
otimizadores_hub = []
losses_hub = []
displays = []

otimizadores = [
    # O.Adam,
    # O.Adamax,
    # O.Adagrad,
    O.ASGD
    # O.SGD
]
loss = [
    # F.l1_loss,
    # F.mse_loss,
    # F.huber_loss,
    F.smooth_l1_loss
]

O_index = -1
L_index = -1


os.system('cls')
n_modelos_ocultos = 700
print(f"Gerando {n_modelos_ocultos} modelo{"s" if n_modelos_ocultos != 1 else ""} oculto{"s" if n_modelos_ocultos != 1 else ""}:")
for i in range(n_modelos_ocultos):
    ambiente = TetrizEnv()
    modelo = DQN(ambiente.observation_space.shape[0], ambiente.action_space.n)
    modelo = modelo.to(device)
    O_index = (O_index + 1) % len(otimizadores)
    if O_index == 0:
        L_index = (L_index + 1) % len(loss)

    modelos.append(modelo)
    ambientes.append(ambiente)
    otimizadores_hub.append(otimizadores[O_index])
    losses_hub.append(loss[L_index])
    displays.append(False)
    print(f'\r{i+1}', end="")

print("Gerando modelo exibido")
ambiente = TetrizEnv(janela=True)
modelo = DQN(ambiente.observation_space.shape[0], ambiente.action_space.n)
modelo = modelo.to(device)
O_index = (O_index + 1) % len(otimizadores)
if O_index == 0:
    L_index = (L_index + 1) % len(loss)

modelos.append(modelo)
ambientes.append(ambiente)
otimizadores_hub.append(otimizadores[O_index])
losses_hub.append(loss[L_index])
displays.append(True)

hub = AI_Hub(
    modelos,
    otimizadores_hub,
    ambientes,
    losses_hub,
    displays
)

os.system('cls')
hub.log()
hub.start()