import random

import numpy as np

import pecas
import pygame, time
import engine
from valores import *
from tabuleiro import Tabuleiro
from operacoes import matriz_to_string

class TetriZzz_Otimizado:
  def __init__(self, x_i, x_f, y_i, y_f, janela):
    self.clock = pygame.time.Clock()
    self.altura_blocos = 16
    self.base_blocos = 10
    self.perdeu = False

    self.tabuleiro = Tabuleiro(self.base_blocos, self.altura_blocos)
    self.engine = engine.Engine(x_i, x_f, y_i, y_f, self.base_blocos, self.altura_blocos, janela)

  def set_tamanho(self, largura_tela, altura_tela, janela):
    self.engine = engine.Engine(largura_tela, altura_tela, self.base_blocos, self.altura_blocos, janela)

  def acao(self, acao):
    self.clock.tick(30)
    match acao:
      case 0: return self.w()
      case 1: return self.a()
      case 2: return self.s()
      case 3: return self.d()
      case 4: return self.espaco()

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

    recompensa_final = (recompensa + recompensa_reiniciou - ((np.std(alturas) - np.std(alturas_anteriores))*peso))

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

    recompensa_final = (recompensa + recompensa_reiniciou - ((np.std(alturas) - np.std(alturas_anteriores))*peso))

    return recompensa_final


  def remover_peca_atual(self):
    coordenadas = self.tabuleiro.coordenadas()
    for coordenada in coordenadas:
      self.engine.remover_bloco(coordenada)
    self.engine.update()

  def desenhar_peca_atual(self):
    coordenadas = self.tabuleiro.coordenadas()
    for coordenada in coordenadas:
      self.engine.desenhar_bloco(coordenada, self.tabuleiro.peca_atual.id)
    self.engine.desenhar_ponto()
    self.engine.update()

  def redesenhar_tudo(self):
    for l in range(self.altura_blocos):
      for c in range(self.base_blocos):
        self.engine.desenhar_bloco([l, c], self.tabuleiro.tabuleiro[l+self.tabuleiro.espaco_adicional_cima][c+self.tabuleiro.tamanho_borda])
    self.engine.desenhar_ponto()
    self.engine.update()