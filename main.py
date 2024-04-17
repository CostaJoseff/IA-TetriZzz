import sys
import threading
import numpy as np
import pygame
from tetriZzz_Otimizado import TetriZzz_Otimizado
import tensorflow as tf
import keras


def criar_janela():
  pygame.init()
  info = pygame.display.Info()
  pygame.display.set_caption("TetriZzz")
  return pygame.display.set_mode([info.current_w, info.current_h])


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

def treinar_modelo(modelo, jogo, resultados):
  recompensa_total = 0
  jogo.redesenhar_tudo()
  passo = 0
  print("thread executando 1")
  while 1:
    recompensa = 0
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return sys.exit()
    if passo % 8 == 0:
      recompensa += jogo.acao(2)
      passo = 0

    print("thread executando 2")
    entrada = ultimas_jogadas.copy()
    entrada.append(jogo.tabuleiro.peca_atual.id)
    entrada.append(jogo.tabuleiro.peca_atual.rotacao)
    entrada = np.append(jogo.tabuleiro.tabuleiro.reshape(-1,1), np.array(entrada))
    acoes = modelo(np.expand_dims(entrada, axis=0))
    acao = tf.argmax(acoes, axis=1).numpy()[0]
    recompensa += jogo.acao(acao)
    recompensa_total += recompensa

    print(acao)

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


def gerar_modelo_alterado():
  inputs = tf.keras.layers.Input(shape=(entrada))
  modelo = tf.keras.layers.Flatten()(inputs)
  modelo = tf.keras.layers.Dense(64, activation='relu')(modelo)
  out = tf.keras.layers.Dense(saida, activation='sigmoid', dtype=tf.float16)(modelo)
  modelo = tf.keras.models.Model(inputs, out)
  modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['loss'])

  return modelo

def escolher_melhores(resultados):
  melhores = []
  for modelo, jogo, recompensa in resultados:
    if len(melhores) < total_escolhidos:
      melhores.append((modelo, jogo, recompensa))
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
        melhores.append((modelo, jogo, recompensa))

  print('Melhores: ', end='')
  for _, _, recompensa in melhores:
    print(f'{recompensa} : ', end='')
  print('\n')
  return melhores

def aplicar_mutacao(modelos, melhores):
  tela = 0
  for modelo, jogo, _ in melhores:
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
      jogo = TetriZzz_Otimizado(i * largura_dos_jogos, (i + 1) * largura_dos_jogos, l * altura_dos_jogos, (l + 1) * altura_dos_jogos, janela)
      modelos.append((novo_modelo, jogo))
      tela += 1


def treinar(modelos, num_episodes):
  for epsodio in range(num_episodes):
    resultados = []
    threads = []
    for modelo, jogo in modelos:
      print("threads vão iniciar")
      thread = threading.Thread(target=treinar_modelo, args=(modelo, jogo, resultados))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    print("join")

    print(f'Epoca: {epsodio}')
    melhores = escolher_melhores(resultados)

    jogo = TetriZzz_Otimizado(i * largura_dos_jogos, (i + 1) * largura_dos_jogos, l * altura_dos_jogos, (l + 1) * altura_dos_jogos, janela)
    modelos = [(melhores[0][0], jogo)]
    aplicar_mutacao(modelos, melhores)

  return modelos

janela = criar_janela()
saida = 5
modelos_totais = 16
total_escolhidos = 2
valor_de_mutacao = 0.2
ultimas_jogadas = [-1, -1, -1, -1]
colunas = 8
linhas = modelos_totais % colunas
if linhas == 0: linhas = modelos_totais / colunas
else: linhas = int(modelos_totais / colunas) + 1
info = pygame.display.Info()
largura_dos_jogos = info.current_w/colunas
altura_dos_jogos = info.current_h/linhas

jogo = TetriZzz_Otimizado(0*largura_dos_jogos, (0+1)*largura_dos_jogos, 0*altura_dos_jogos, (0+1)*altura_dos_jogos, janela)
entrada = ultimas_jogadas.copy()
entrada.append(jogo.tabuleiro.peca_atual.id)
entrada.append(jogo.tabuleiro.peca_atual.rotacao)
entrada = np.append(jogo.tabuleiro.tabuleiro.reshape(-1, 1), np.array(entrada))
entrada = entrada.shape

modelos = []
for l in range(int(linhas)):
  for i in range(int(colunas)):
    print(f'x_i {i*largura_dos_jogos} x_f {(i+1)*largura_dos_jogos} y_i {l*altura_dos_jogos} y_f {(l+1)*altura_dos_jogos}')
    modelo = gerar_modelo_alterado()
    jogo = TetriZzz_Otimizado(i*largura_dos_jogos, (i+1)*largura_dos_jogos, l*altura_dos_jogos, (l+1)*altura_dos_jogos, janela)
    modelos.append((modelo, jogo))
    print(f'Modelo {len(modelos)} criado')

modelos = treinar(modelos, 100)
