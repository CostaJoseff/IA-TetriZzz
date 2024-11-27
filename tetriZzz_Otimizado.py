import random

import numpy as np

import pecas
import pygame, time
import engine
from valores import *
from tabuleiro import Tabuleiro
from operacoes import matriz_to_string

class TetriZzz_Otimizado:
  def __init__(self, x_i=None, x_f=None, y_i=None, y_f=None, janela=None, desenhar=True):
    self.clock = pygame.time.Clock()
    self.altura_blocos = 16
    self.base_blocos = 10
    self.perdeu = False

    self.tabuleiro = Tabuleiro(self.base_blocos, self.altura_blocos)
    if desenhar:
      self.engine = engine.Engine(x_i, x_f, y_i, y_f, self.base_blocos, self.altura_blocos, janela, self.tabuleiro)
    else:
      self.engine = None

  def set_tamanho(self, largura_tela, altura_tela, janela):
    self.engine = engine.Engine(largura_tela, altura_tela, self.base_blocos, self.altura_blocos, janela, self.tabuleiro)

  def acao(self, acao):
    self.clock.tick(30)
    match acao:
      case 0: 
        recom = self.w()
        self.tabuleiro.pontos += recom
        return recom
      case 1: 
        recom = self.a()
        self.tabuleiro.pontos += recom
        return recom
      case 2: 
        recom = self.s()
        self.tabuleiro.pontos += recom
        return recom
      case 3: 
        recom = self.d()
        self.tabuleiro.pontos += recom
        return recom
      case 4: 
        recom = self.espaco()
        self.tabuleiro.pontos += recom
        return recom
      
  def step(self, acao):
    self.clock.tick(30)
    match acao:
      case 0: 
        recom = self.w()
        self.tabuleiro.pontos += recom
        return self.tabuleiro.tabuleiro, recom, self.perdeu
      case 1: 
        recom = self.a()
        self.tabuleiro.pontos += recom
        return self.tabuleiro.tabuleiro, recom, self.perdeu
      case 2: 
        recom = self.s()
        self.tabuleiro.pontos += recom
        return self.tabuleiro.tabuleiro, recom, self.perdeu
      case 3: 
        recom = self.d()
        self.tabuleiro.pontos += recom
        return self.tabuleiro.tabuleiro, recom, self.perdeu
      case 4: 
        recom = self.espaco()
        self.tabuleiro.pontos += recom
        return self.tabuleiro.tabuleiro, recom, self.perdeu

  def a(self):
    self.remover_peca_atual()
    recompensa = self.tabuleiro.mover_para_esquerda()
    self.desenhar_peca_atual()
    return recompensa

  def w(self):
    self.remover_peca_atual()
    recompensa = self.tabuleiro.rotacionar_peca()
    self.desenhar_peca_atual()
    return recompensa

  def d(self):
    self.remover_peca_atual()
    recompensa = self.tabuleiro.mover_para_direita()
    self.desenhar_peca_atual()
    return recompensa

  def espaco(self):
    self.remover_peca_atual()
    recompensa, alturas, alturas_anteriores = self.tabuleiro.ate_o_final()
    self.desenhar_peca_atual()
    recompensa_reiniciou = self.tabuleiro.reiniciou()
    self.desenhar_peca_atual()
    if recompensa_reiniciou == punicao_perdeu:
      self.perdeu = True
      self.redesenhar_tudo()
    elif recompensa_reiniciou > 0:
      self.redesenhar_tudo()

    # print("\n\n"+str(np.std(alturas) - np.std(alturas_anteriores)))
    # print((recompensa + recompensa_reiniciou) - (np.std(alturas) - np.std(alturas_anteriores))*peso)
    # time.sleep(1)
    penali = self.calcular_penalidade(alturas, alturas_anteriores)
    # print(f"Penali = {penali}")
    recompensa_final = (recompensa + recompensa_reiniciou) - penali
    # print(f"Recomp {recompensa_final}")
    # input()
    return recompensa_final

  def s(self):
    self.remover_peca_atual()
    recompensa, alturas, alturas_anteriores = self.tabuleiro.mover_para_baixo()
    if recompensa == nenhuma_recompensa:
      self.desenhar_peca_atual()
      return recompensa

    self.desenhar_peca_atual()
    recompensa_reiniciou = self.tabuleiro.reiniciou()
    self.desenhar_peca_atual()
    if recompensa_reiniciou == punicao_perdeu:
      self.perdeu = True
      self.redesenhar_tudo()
    elif recompensa_reiniciou > 0:
      self.redesenhar_tudo()

    # print("\n\n"+str(np.std(alturas) - np.std(alturas_anteriores)))
    # print((recompensa + recompensa_reiniciou) - (np.std(alturas) - np.std(alturas_anteriores))*peso)
    # time.sleep(1)
    penali = self.calcular_penalidade(alturas, alturas_anteriores)
    # print(f"Penali = {penali}")
    recompensa_final = (recompensa + recompensa_reiniciou) - penali
    # print(f"Recomp {recompensa_final}\n")

    # input()
    return recompensa_final
  
  def calcular_penalidade(self, alturas_atual, alturas_anteriores):
    # 1. Calculando o incremento nas alturas
    incremento = np.array(alturas_atual) - np.array(alturas_anteriores)

    # Recompensa por decremento de altura (se altura diminuiu)
    recompensa_decremento = np.sum(np.minimum(incremento, 0))  # Recompensa se a altura diminuiu
    
    # 2. Penalidade por diferença de altura entre colunas vizinhas
    diffs = np.abs(np.diff(alturas_atual))
    diffs_anter = np.abs(np.diff(alturas_anteriores))
    penalidade_dif = np.sum(diffs)  # Penaliza diferenças grandes de altura entre colunas vizinhas
    peso_penali_diff = 0.7
    if np.sum(diffs) <= np.sum(diffs_anter) or np.std(alturas_atual) - np.std(alturas_anteriores) <= 0.2:
      penalidade_dif = 0

    # 3. Desvio padrão das alturas para medir a dispersão
    desvio_padrao = np.std(alturas_atual)
    
    # 4. Penaliza excessos de altura em colunas muito altas
    if max(alturas_atual) > max(alturas_anteriores):
      penalidade_excesso = np.sum([h for h in alturas_atual if h > 10])  # Ajuste o limite conforme necessário
    else:
      penalidade_excesso = 0

    # 7. Penalidade extra por diferença entre estados atual e anterior (verifica se as alturas aumentaram ou diminuíram)
    penalidade_diff_estado = np.sum(np.abs(incremento))  # Penaliza mudanças muito bruscas nas alturas

    # print(f"decrem {recompensa_decremento*4}")
    # print(f"diffs {penalidade_dif*peso_penali_diff}")
    # print(f"stdr {desvio_padrao*1.7}")
    # print(f"Excess {penalidade_excesso*1.3}")
    # print(f"diff temp {penalidade_diff_estado*1.8}")
    # A penalidade total leva em conta:
    penalidade_total = recompensa_decremento * 4 + \
                        penalidade_dif * peso_penali_diff + \
                        desvio_padrao * 1.7 + \
                        penalidade_excesso * 1.3 + \
                        penalidade_diff_estado * 1.8

    return penalidade_total

  def remover_peca_atual(self):
    coordenadas = self.tabuleiro.coordenadas()
    if self.engine is not None:
      for coordenada in coordenadas:
        self.engine.remover_bloco(coordenada)
      self.engine.update()

  def desenhar_peca_atual(self):
    coordenadas = self.tabuleiro.coordenadas()
    if self.engine is not None:
      for coordenada in coordenadas:
        self.engine.desenhar_bloco(coordenada, self.tabuleiro.peca_atual.id)
      self.engine.desenhar_ponto()
      self.engine.update()

  def redesenhar_tudo(self):
    if self.engine is not None:
      for l in range(self.altura_blocos):
        for c in range(self.base_blocos):
          self.engine.desenhar_bloco([l, c], self.tabuleiro.tabuleiro[l+self.tabuleiro.espaco_adicional_cima][c+self.tabuleiro.tamanho_borda])
      self.engine.desenhar_ponto()
      self.engine.update()