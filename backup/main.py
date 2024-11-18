import sys
import threading
import numpy as np
import pygame
from tetriZzz_Otimizado import TetriZzz_Otimizado
import tensorflow as tf
import random as rd
import os
from tensorflow.keras import layers, Model


def criar_janela():
  os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (900, 50)
  pygame.init()
  info = pygame.display.Info()
  pygame.display.set_caption("TetriZzz")
  return pygame.display.set_mode([info.current_w/2, info.current_h])


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

def treinar_bot(bot, jogo, resultados):
  recompensa_total = 0
  jogo.redesenhar_tudo()
  passo = 0
  while 1:
    recompensa = 0
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return sys.exit()
    if passo % 8 == 0:
      recompensa += jogo.acao(2)
      passo = 0

    estado = ultimas_jogadas.copy()
    for altura in jogo.tabuleiro.calcular_alturas():
      estado.append(altura)
    estado.append(jogo.tabuleiro.peca_atual.id)
    estado.append(jogo.tabuleiro.peca_atual.rotacao)
    estado.append(jogo.tabuleiro.linha_atual)
    estado.append(jogo.tabuleiro.coluna_atual)

    estado = np.array(estado)

    acoes = bot(np.expand_dims(estado, axis=0))
    acao = tf.argmax(acoes, axis=1).numpy()[0]
    recompensa += jogo.acao(acao)
    recompensa_total += recompensa
    #print(recompensa)

    for i in range(len(ultimas_jogadas)-1, 0, -1):
      ultimas_jogadas[i] = ultimas_jogadas[i-1]
    ultimas_jogadas[0] = acao

    if recompensa != 0:
      with tf.GradientTape() as tape:
        acoes = bot(np.expand_dims(estado, axis=0))
        index = tf.argmax(acoes, axis=1).numpy()[0]
        valor = tf.gather(acoes, index, axis=1)
        valor = valor * recompensa
        loss = tf.reduce_mean(valor)
      grads = tape.gradient(loss, bot.trainable_variables)
      bot.optimizer.apply_gradients(zip(grads, bot.trainable_variables))

    if jogo.perdeu: break
    passo += 1

  resultados.append((bot, jogo, recompensa_total))

def gerar_camada_input():
  entrada = ultimas_jogadas.copy()
  for altura in jogo.tabuleiro.calcular_alturas():
    entrada.append(altura)
  entrada.append(jogo.tabuleiro.peca_atual.id)
  entrada.append(jogo.tabuleiro.peca_atual.rotacao)
  entrada.append(jogo.tabuleiro.linha_atual)
  entrada.append(jogo.tabuleiro.coluna_atual)

  return np.array(entrada)

def gerar_bot():
  entrada = gerar_camada_input()

  inputs = tf.keras.layers.Input(shape=entrada.shape)
  hidden = tf.keras.layers.Dense(9, activation='relu')(inputs)
  hidden = tf.keras.layers.Dropout(0.25)(hidden)
  hidden = tf.keras.layers.Dense(5, activation='relu')(hidden)
  out = tf.keras.layers.Dense(saida, activation='sigmoid', dtype=tf.float16)(hidden)

  bot = tf.keras.models.Model(inputs, out)
  bot.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['loss'])

  return bot

def gerar_bot_modificado(bot):
  pesos_da_rede = bot.get_weights()
  pesos_perturbados = [peso + (((-1)**(rd.randint(1, 2)))*np.random.normal(scale=mutacao, size=peso.shape)) for peso in pesos_da_rede]

  entrada = gerar_camada_input()

  inputs = tf.keras.layers.Input(shape=entrada.shape)
  hidden = tf.keras.layers.Dense(9, activation='relu')(inputs)
  hidden = tf.keras.layers.Dropout(0.25)(hidden)
  hidden = tf.keras.layers.Dense(5, activation='relu')(hidden)
  out = tf.keras.layers.Dense(saida, activation='sigmoid', dtype=tf.float16)(hidden)

  novo_bot = tf.keras.models.Model(inputs, out)
  novo_bot.set_weights(pesos_perturbados)
  novo_bot.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return novo_bot

def escolher_melhores(resultados):
  melhores = []
  for bot, jogo, recompensa in resultados:
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

def treinar(bots, num_episodes):
  for epsodio in range(num_episodes):
    resultados = []
    threads = []
    for bot, jogo in bots:
      thread = threading.Thread(target=treinar_bot, args=(bot, jogo, resultados))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()


    print(f'Epoca: {epsodio}')
    melhores = escolher_melhores(resultados)
    bots = aplicar_mutacao(melhores)
    bots = adicionar_jogos(bots)

  return bots

janela = criar_janela()
saida = 5
total_de_bots = 32
total_escolhidos = 1
mutacao = 0.05
ultimas_jogadas = [-1, -1, -1, -1]
colunas = 8
if total_de_bots < colunas:
  colunas = total_de_bots
linhas = total_de_bots % colunas
if linhas == 0: linhas = total_de_bots / colunas
else: linhas = int(total_de_bots / colunas) + 1
info = pygame.display.Info()
largura_dos_jogos = (info.current_w)/colunas
altura_dos_jogos = (info.current_h)/linhas

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
bots = treinar(bots, 10000)
