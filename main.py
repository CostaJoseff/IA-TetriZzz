import sys
import threading
from valores import punicao_perdeu

import numpy as np
import pygame
from tetriZzz_Otimizado import TetriZzz_Otimizado
import tensorflow as tf


def criar_janela(largura, altura):
  pygame.init()
  pygame.display.set_caption("TetriZzz")
  return pygame.display.set_mode([largura, altura])


# janela = criar_janela(800, 600)
# jogo = TetriZzz_Otimizado(800, 600, janela)
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


aaa = 350
janela = criar_janela(800, aaa)

def treinar_modelo(modelo, jogo, max_steps_por_epsod, resultados):
  recompensa_total = 0
  ultimas_jogadas = [-1, -1, -1, -1]
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

    entrada = ultimas_jogadas.copy()
    entrada.append(jogo.tabuleiro.peca_atual.id)
    entrada.append(jogo.tabuleiro.peca_atual.rotacao)
    entrada = np.append(jogo.tabuleiro.tabuleiro.reshape(-1,1), np.array(entrada))
    acoes = modelo(np.expand_dims(entrada, axis=0))
    acao = tf.argmax(acoes, axis=1).numpy()[0]
    recompensa += jogo.acao(acao)
    recompensa_total += recompensa

    for i in range(len(ultimas_jogadas)-1, 0, -1):
      ultimas_jogadas[i] = ultimas_jogadas[i-1]
    ultimas_jogadas[0] = acao

    with tf.GradientTape() as tape:
      acoes = modelo(np.expand_dims(entrada, axis=0))
      index = tf.argmax(acoes, axis=1).numpy()[0]
      valor = tf.gather(acoes, index, axis=1)
      valor = valor * recompensa
      loss = tf.reduce_mean(-valor)
    grads = tape.gradient(loss, modelo.trainable_variables)
    modelo.optimizer.apply_gradients(zip(grads, modelo.trainable_variables))

    if jogo.perdeu: break
    passo += 1

  resultados.append((modelo, jogo, recompensa_total))

def escolher_2_melhores(resultados):
  melhores_2 = []
  for modelo, jogo, recompensa in resultados:
    print(recompensa, end=" || ")
    if len(melhores_2) < 1:
      melhores_2.append((modelo, jogo, recompensa))
    else:
      alvo = (None, 0)
      for modelo_selecionado, jogo_selecionado, recompensa_selecionada in melhores_2:
        if recompensa_selecionada < recompensa:
          if alvo[0] == None:
            alvo = (modelo_selecionado, jogo_selecionado, recompensa_selecionada)
          elif alvo[2] > recompensa_selecionada:
            alvo = (modelo_selecionado, jogo_selecionado, recompensa_selecionada)

      if alvo[0] != None:
        melhores_2.remove(alvo)
        melhores_2.append((modelo, jogo, recompensa))

  print('')
  return melhores_2

def aplicar_mutacao(modelos, melhores_2):
  tela = 2
  for modelo, jogo, _ in melhores_2:
    for i in range(7):
      pesos_da_rede = modelo.get_weights()
      perturbacao = 0.1
      pesos_perturbados = [peso - np.random.normal(scale=perturbacao, size=peso.shape) for peso in pesos_da_rede]

      entrada = ultimas_jogadas.copy()
      entrada.append(jogo.tabuleiro.peca_atual.id)
      entrada.append(jogo.tabuleiro.peca_atual.rotacao)
      entrada = np.append(jogo.tabuleiro.tabuleiro.reshape(-1,1), np.array(entrada))

      inputs = tf.keras.layers.Input(shape=(entrada.shape))
      novo_modelo = tf.keras.layers.Flatten()(inputs)
      novo_modelo = tf.keras.layers.Dense(64, activation='relu')(novo_modelo)
      out = tf.keras.layers.Dense(saida, activation='sigmoid', dtype=tf.float16)(novo_modelo)
      novo_modelo = tf.keras.models.Model(inputs, out)
      novo_modelo.set_weights(pesos_perturbados)
      novo_modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      modelos.append((novo_modelo, TetriZzz_Otimizado(tela * 100, aaa, janela)))
      tela += 1


def treinar(modelos, num_episodes, max_steps_por_epsod):
  for epsodio in range(num_episodes):
    resultados = []
    threads = []
    for modelo, jogo in modelos:
      thread = threading.Thread(target=treinar_modelo, args=(modelo, jogo, max_steps_por_epsod, resultados))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    melhores_2 = escolher_2_melhores(resultados)

    print(f'Epoca: {epsodio}')
    print(f'Top: {melhores_2[0][2]}')
    print("-" * 10)

    modelos = [(melhores_2[0][0], TetriZzz_Otimizado(100, aaa, janela))]
    aplicar_mutacao(modelos, melhores_2)

  return modelos

saida = 5
modelos_totais = 8


modelos = []
for i in range(modelos_totais):
  ultimas_jogadas = [-1, -1, -1, -1]
  jogo = TetriZzz_Otimizado(100*(i+1), aaa, janela)
  entrada = ultimas_jogadas.copy()
  entrada.append(jogo.tabuleiro.peca_atual.id)
  entrada.append(jogo.tabuleiro.peca_atual.rotacao)
  entrada = np.append(jogo.tabuleiro.tabuleiro.reshape(-1,1), np.array(entrada))

  inputs = tf.keras.layers.Input(shape=(entrada.shape))
  modelo = tf.keras.layers.Flatten()(inputs)
  modelo = tf.keras.layers.Dense(64, activation='relu')(modelo)
  out = tf.keras.layers.Dense(saida, activation='sigmoid', dtype=tf.float16)(modelo)
  modelo = tf.keras.models.Model(inputs, out)
  modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['loss'])
  modelos.append((modelo, jogo))

modelos, jogo = treinar(modelos,10000, 100)
