import random
import pecas
import pygame, time
import engine
from tabuleiro import Tabuleiro
from operacoes import matriz_to_string

class TetriZzz_Otimizado:
  def __init__(self):
    multiplicador = 90
    self.altura_blocos = 16
    self.base_blocos = 10
    largura_tela = 5 * multiplicador
    altura_tela = 8 * multiplicador

    self.tabuleiro = Tabuleiro(self.base_blocos, self.altura_blocos)
    self.engine = engine.Engine(largura_tela, altura_tela, self.base_blocos, self.altura_blocos)

  def a(self):
    self.remover_peca_atual()
    retorno = self.tabuleiro.mover_para_esquerda()
    self.desenhar_peca_atual()
    time.sleep(0.01)
    return retorno

  def w(self):
    self.remover_peca_atual()
    retorno = self.tabuleiro.rotacionar_peca()
    self.desenhar_peca_atual()
    time.sleep(0.01)
    return retorno

  def d(self):
    self.remover_peca_atual()
    retorno = self.tabuleiro.mover_para_direita()
    self.desenhar_peca_atual()
    time.sleep(0.01)
    return retorno

  def espaco(self):
    self.remover_peca_atual()
    retorno = self.tabuleiro.ate_o_final()
    self.desenhar_peca_atual()
    if retorno == 0:
      retorno = self.tabuleiro.reiniciou()
      if retorno == -5:
        self.engine.limpar()
    time.sleep(0.01)
    return retorno

  def s(self):
    self.remover_peca_atual()
    retorno = self.tabuleiro.mover_para_baixo()
    self.desenhar_peca_atual()
    if retorno == 0:
      retorno = self.tabuleiro.reiniciou()
      if retorno == -5:
        self.engine.limpar()
    time.sleep(0.01)
    return retorno

  def remover_peca_atual(self):
    coordenadas = self.tabuleiro.coordenadas()
    for coordenada in coordenadas:
      self.engine.remover_bloco(coordenada)
    self.engine.update()

  def desenhar_peca_atual(self):
    coordenadas = self.tabuleiro.coordenadas()
    for coordenada in coordenadas:
      self.engine.desenhar_bloco(coordenada, self.tabuleiro.peca_atual.id)
    self.engine.update()