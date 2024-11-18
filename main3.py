import sys
import threading
import numpy as np
import pygame

from tetriZzz_Otimizado import TetriZzz_Otimizado
import tensorflow as tf
import random as rd
import os

def criar_janela():
  os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (900, 50)
  pygame.init()
  info = pygame.display.Info()
  pygame.display.set_caption("TetriZzz")
  return pygame.display.set_mode([info.current_w/2, info.current_h])

def gerar_bot():
  entrada = jogo.tabuleiro.tabuleiro
  inputs = tf.keras.layers.Input(shape=(entrada.shape[0], entrada.shape[1], 1))
  hidden = tf.keras.layers.Conv2D(32, (4, 4), activation='relu')(inputs)
  hidden = tf.keras.layers.Dense(9, activation='relu')(inputs)
  hidden = tf.keras.layers.Dropout(0.25)(hidden)
  hidden = tf.keras.layers.Dense(5, activation='relu')(hidden)
  out = tf.keras.layers.Dense(camada_de_saida, activation='softmax')(hidden)

  bot = tf.keras.models.Model(inputs, out)
  bot.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return bot

def gerar_bot_modificado(bot):
  pesos_da_rede = bot.get_weights()
  pesos_perturbados = [peso + (((-1)**(rd.randint(1, 2)))*np.random.normal(scale=mutacao, size=peso.shape)) for peso in pesos_da_rede]

  entrada = jogo.tabuleiro.tabuleiro
  inputs = tf.keras.layers.Input(shape=(entrada.shape[0], entrada.shape[1], 1))
  hidden = tf.keras.layers.Conv2D(32, (4, 4), activation='relu')(inputs)
  hidden = tf.keras.layers.Dense(9, activation='relu')(inputs)
  hidden = tf.keras.layers.Dropout(0.25)(hidden)
  hidden = tf.keras.layers.Dense(5, activation='relu')(hidden)
  out = tf.keras.layers.Dense(camada_de_saida, activation='softmax')(hidden)

  novo_bot = tf.keras.models.Model(inputs, out)
  novo_bot.set_weights(pesos_perturbados)
  novo_bot.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return novo_bot

def escolher_melhores(bots):
  melhores = []
  jogo: TetriZzz_Otimizado
  for bot, jogo in bots:
    recompensa = jogo.tabuleiro.pontos
    if len(melhores) < total_escolhidos:
      melhores.append((bot, jogo, recompensa))
    else:
      alvo = (None, 0)
      for modelo_selecionado, jogo_selecionado, recompensa_selecionada in melhores:
        if recompensa_selecionada < recompensa:
          if alvo[0] == None:
            alvo = (modelo_selecionado, jogo_selecionado, recompensa_selecionada)
          elif alvo[2] > recompensa_selecionada:
            alvo = (modelo_selecionado, jogo_selecionado, recompensa_selecionada)

      if alvo[0] != None:
        melhores.remove(alvo)
        melhores.append((bot, jogo, recompensa))

  print('Melhores: ', end='')
  for _, _, recompensa in melhores:
    print(f'{recompensa} : ', end='')
  print('\n')
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
      jogo = TetriZzz_Otimizado(i * largura_dos_jogos, (i + 1) * largura_dos_jogos, l * altura_dos_jogos, (l + 1) * altura_dos_jogos, janela)
      retorno.append((bots[len(retorno)], jogo))
      if len(retorno) == total_de_bots:
        break
    if len(retorno) == total_de_bots:
      break
  return retorno

def deixar_jogar(bot, jogo: TetriZzz_Otimizado):
    jogo.redesenhar_tudo()
    while 1:
      estado = jogo.tabuleiro.tabuleiro
      estado = estado[..., np.newaxis]

      if i % 8 == 0:
        acao = 2
      else:
        acao = bot.predict(estado)
        acao = acao.argmax(axis=1)

      jogo.acao(acao)
      if jogo.perdeu: break
    

def treinar(bots, epsod):
  for ep in range(epsod):
    threads = []
    for bot, jogo in bots:
      thread = threading.Thread(target=deixar_jogar, args=(bot, jogo))
      thread.start()
      threads.append(thread)
    
    for thread in threads:
      thread.join()

    n_melhores = escolher_melhores(bots)
    bots = aplicar_mutacao(n_melhores)
    bots = adicionar_jogos(bots)
  
  return bots

janela = criar_janela()
camada_de_saida = 5
total_de_bots = 3
total_escolhidos = 1
mutacao = 0.05
colunas = 3
if total_de_bots < colunas:
  colunas = total_de_bots
linhas = total_de_bots % colunas
if linhas == 0: linhas = total_de_bots / colunas
else: linhas = int(total_de_bots / colunas) + 1
info = pygame.display.Info()
largura_dos_jogos = (info.current_w-300)/colunas
altura_dos_jogos = (info.current_h-300)/linhas

jogo = TetriZzz_Otimizado(0*largura_dos_jogos, (0+1)*largura_dos_jogos, 0*altura_dos_jogos, (0+1)*altura_dos_jogos, janela)

bots = []
for l in range(int(linhas)):
  for i in range(int(colunas)):
    bot = gerar_bot()
    jogo = TetriZzz_Otimizado(i*largura_dos_jogos, (i+1)*largura_dos_jogos, l*altura_dos_jogos, (l+1)*altura_dos_jogos, janela)
    bots.append((bot, jogo))
    print(f'Modelo {len(bots)} criado')
    if len(bots) == total_de_bots:
      break
  if len(bots) == total_de_bots:
    break

bots = treinar(bots, 100000) 