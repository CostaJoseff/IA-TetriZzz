import math
import sys
import threading
import time
import numpy as np
import pygame
from tetriZzz_Otimizado import TetriZzz_Otimizado
import tensorflow as tf
import random as rd
import os
import torch.nn as nn
import torch 
from matplotlib import pyplot as plt
import cv2

class ModeloBot(nn.Module):
        def __init__(self):
            super(ModeloBot, self).__init__()
            self.fc1 = nn.Linear(25 * 18, 90)  # Flatten + Dense(9)
            self.fc2 = nn.Linear(90, 512) 
            self.fc3 = nn.Linear(512, 512)       # Dense(5)
            self.fc4 = nn.Linear(512, saida)    # Dense(saida)
            self.dropout = nn.Dropout(0.25)   # Dropout
        
        def forward(self, x):
            x = x.view(1, 25 * 18)           # Flatten
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

def criar_janela():
  os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (0, 0)
  pygame.init()
  info = pygame.display.Info()
  pygame.display.set_caption("TetriZzz")
  return pygame.display.set_mode([info.current_w-300, info.current_h-500])


# janela = criar_janela()
# info = pygame.display.Info()
# jogo = TetriZzz_Otimizado(0, info.current_w, 0, info.current_h, janela)
# recompensa = 0
# while True:
#   for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#       sys.exit()
#     elif event.type == pygame.KEYDOWN:
#       if event.key == pygame.K_a:
#         recompensa = jogo.a()
#       elif event.key == pygame.K_d:
#         recompensa = jogo.d()
#       elif event.key == pygame.K_w:
#         recompensa = jogo.w()
#       elif event.key == pygame.K_SPACE:
#         recompensa = jogo.espaco()
#       elif event.key == pygame.K_s:
#         recompensa = jogo.s()
#
#   print(recompensa)

def treinar_bot(bot, jogo: TetriZzz_Otimizado, resultados):
  jogo.redesenhar_tudo()
  passo = 0
  while 1:
    pygame.event.pump()
    if passo % 8 == 0:
      jogo.acao(2)
      passo = 0

    estado = jogo.tabuleiro.tabuleiro
    estado = estado[None, ...]
    estado = torch.tensor(estado, dtype=torch.float32)

    with torch.no_grad():
      acoes = bot(estado)
    acao = torch.argmax(acoes, dim=1).item()
    jogo.acao(acao)

    if jogo.perdeu: break
    passo += 1

  resultados.append((bot, jogo))

def gerar_bot():
  novo_bot = ModeloBot()
  return novo_bot

def gerar_bot_modificado(bot):
  pesos_da_rede = [param.detach().cpu().numpy() for param in bot.parameters()]
  pesos_perturbados = [peso * ((-1)**(rd.randint(1, 2)))*mutacao for peso in pesos_da_rede]
  pesos_perturbados = []
  for peso in pesos_da_rede:
    if rd.randint(0, 1) == 1:
      pesos_perturbados.append(
        peso + (peso * ((-1)**(rd.randint(1, 2)))*mutacao)
      )
    else:
      pesos_perturbados.append(peso)

  novo_bot = ModeloBot()

  with torch.no_grad():
      params_iter = iter(novo_bot.parameters())
      for peso_perturbado in pesos_perturbados:
          parametro = next(params_iter)
          parametro.copy_(torch.tensor(peso_perturbado, dtype=parametro.dtype))

  return novo_bot

def escolher_melhores(resultados):
  melhores = []
  jogo: TetriZzz_Otimizado
  resultados = [[bot, jogo, jogo.tabuleiro.pontos] for bot, jogo in resultados]
  resultados.sort(key=lambda x: x[2], reverse=True)
  melhores = [resultados[i] for i in range(total_escolhidos)]

  recomps = [x for _, _, x in melhores]
  print(f"Media {np.mean(recomps)}")
  print(f"Desvio {np.std(recomps)}\n")
  # print('Melhores: ', end='')
  # i=0
  # for _, _, recompensa in melhores:
  #   print(f'{recompensa} : ', end='')
  #   # melhores_n[i].append(recompensa)
  #   i += 1
  # print('\n')
  return melhores

def aplicar_mutacao(melhores):
  bots = []
  for bot, _, _ in melhores:
    bots.append(bot)
    for i in range(int((total_de_bots / total_escolhidos) - 1)):
      novo_bot = gerar_bot_modificado(bot)
      bots.append(novo_bot)
  return bots

def adicionar_jogos(bots):
  retorno = []
  for l in range(int(linhas)):
    for i in range(int(colunas)):
      jogo = TetriZzz_Otimizado(i * largura_dos_jogos, (i + 1) * largura_dos_jogos, l * altura_dos_jogos, (l + 1) * altura_dos_jogos, janela, desenhar=True)
      retorno.append((bots[len(retorno)], jogo))
      if len(retorno) == total_de_bots:
        break
    if len(retorno) == total_de_bots:
      break

  ln = len(retorno)
  for i in range(total_de_bots - ln):
    bot = gerar_bot()
    jogo = TetriZzz_Otimizado(0*largura_dos_jogos, (0+1)*largura_dos_jogos, 0*altura_dos_jogos, (0+1)*altura_dos_jogos, janela, desenhar=False)
    retorno.append((bot, jogo))
  return retorno

def treinar(bots, num_episodes):
  for epsodio in range(num_episodes):
    print(f'Epoca: {epsodio}', end='', flush=True)
    inicio = time.time()
    resultados = []
    threads = []
    for bot, jogo in bots:
      thread = threading.Thread(target=treinar_bot, args=(bot, jogo, resultados))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    fim = time.time()
    tempo = int(fim - inicio)
    print(f" | {tempo} s")
    melhores = escolher_melhores(resultados)
    bots = aplicar_mutacao(melhores)
    bots = adicionar_jogos(bots)

    # ymin = False
    # ymax = False
    # for line, melhor_m in zip(lines, melhores_n):
    #   line.set_ydata(melhor_m)
    #   mini = min(melhor_m)
    #   maxi = max(melhor_m)
    #   ymin = mini if mini < ymin else ymin
    #   ymax = maxi if maxi > ymax else ymax

    # ax.set_ylim(ymin-100, ymax+100)
    # ax.set_xlim(0, len(melhor_m))
    # plt.pause(0.1)
    # plt.draw()
    # plt.pause(0.1)
    # plt.show()

  return bots

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
colunas = 3
linhas = 2
# if total_de_bots < colunas:
#   colunas = total_de_bots
# linhas = total_de_bots % colunas
# if linhas == 0: linhas = total_de_bots / colunas
# else: linhas = int(total_de_bots / colunas) + 1
info = pygame.display.Info()
largura_dos_jogos = (info.current_w)/colunas
altura_dos_jogos = (info.current_h)/linhas

jogo = TetriZzz_Otimizado(0*largura_dos_jogos, (0+1)*largura_dos_jogos, 0*altura_dos_jogos, (0+1)*altura_dos_jogos, janela, desenhar=False)
bots = []
for l in range(int(linhas)):
  for i in range(int(colunas)):
    bot = gerar_bot()
    jogo = TetriZzz_Otimizado(i*largura_dos_jogos, (i+1)*largura_dos_jogos, l*altura_dos_jogos, (l+1)*altura_dos_jogos, janela, desenhar=True)
    bots.append((bot, jogo))
    if len(bots) == total_de_bots:
      break
  if len(bots) == total_de_bots:
    break

ln = len(bots)
for i in range(total_de_bots - ln):
  bot = gerar_bot()
  jogo = TetriZzz_Otimizado(0*largura_dos_jogos, (0+1)*largura_dos_jogos, 0*altura_dos_jogos, (0+1)*altura_dos_jogos, janela, desenhar=False)
  bots.append((bot, jogo))

bots = treinar(bots, 10000)
