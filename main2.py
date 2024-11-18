import sys
import threading
import numpy as np
import pygame

from tetriZzz_Otimizado import TetriZzz_Otimizado
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import random as rd
import os

def criar_janela():
    pygame.init()
    info = pygame.display.Info()
    pygame.display.set_caption("TetriZzz")
    return pygame.display.set_mode([info.current_w/4, info.current_h/2])

def gerar_camada_input(ultimas_jogadas, jogo):
    entrada = ultimas_jogadas.copy()
    for altura in jogo.tabuleiro.calcular_alturas():
        entrada.append(altura)
    entrada.append(jogo.tabuleiro.peca_atual.id)
    entrada.append(jogo.tabuleiro.peca_atual.rotacao)
    entrada.append(jogo.tabuleiro.linha_atual)
    entrada.append(jogo.tabuleiro.coluna_atual)

    return np.array(entrada)

def gerar_modelo(jogo):
    ultimas_jogadas = [-1, -1, -1, -1]
    input = gerar_camada_input(ultimas_jogadas, jogo)
    modelo = Sequential()
    modelo.add(Flatten(input_shape=input.shape))
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dense(5, activation='linear'))
    return modelo

def gerar_agente(modelo):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=30000, window_length=10)
    dqn = DQNAgent(model=modelo, memory=memory, policy=policy, nb_actions=5, nb_steps_warmup=4, target_model_update=1e-2)
    return dqn

janela = criar_janela()
total_de_bots = 1
colunas = 1
linhas = 1
info = pygame.display.Info()
largura_dos_jogos = (info.current_w)/colunas
altura_dos_jogos = (info.current_h)/linhas

jogo = TetriZzz_Otimizado(0*largura_dos_jogos, (0+1)*largura_dos_jogos, 0*altura_dos_jogos, (0+1)*altura_dos_jogos, janela)
modelo = gerar_modelo(jogo)
modelo.summary()
dqn = gerar_agente(modelo)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(jogo, nb_steps=5000, visualize=True, verbose=1)