from Util.Operacoes import compute_punishment
from Jogo.Tabuleiro import Tabuleiro
from Jogo.Engine import Engine
from Jogo.valores import *
import pygame

class TetriZzz_Otimizado:
  def __init__(self, x_i, x_f, y_i, y_f, janela=False):
    self.clock = pygame.time.Clock()
    self.altura_blocos = 16
    self.base_blocos = 10
    self.perdeu = False
    self.fps = 30

    self.tabuleiro = Tabuleiro(self.base_blocos, self.altura_blocos)
    if janela:
      self.engine = Engine(x_i, x_f, y_i, y_f, self.base_blocos, self.altura_blocos, janela, self.tabuleiro)
    else:
      self.engine = None

  def set_tamanho(self, largura_tela, altura_tela, janela):
    self.engine = Engine(largura_tela, altura_tela, self.base_blocos, self.altura_blocos, janela)

  def acao(self, acao):
    self.clock.tick(self.fps)
    if acao == 0:
      return self.w()
    elif acao == 1:
      return self.a()
    elif acao == 2:
      return self.s()
    elif acao == 3:
      return self.d()
    elif acao == 4:
      return self.espaco()


  def a(self):
    self.remover_peca_atual()
    recompensa = self.tabuleiro.mover_para_esquerda()
    self.tabuleiro.pontos += recompensa
    self.desenhar_peca_atual()
    return recompensa

  def w(self):
    self.remover_peca_atual()
    recompensa = self.tabuleiro.rotacionar_peca()
    self.tabuleiro.pontos += recompensa
    self.desenhar_peca_atual()
    return recompensa

  def d(self):
    self.remover_peca_atual()
    recompensa = self.tabuleiro.mover_para_direita()
    self.tabuleiro.pontos += recompensa
    self.desenhar_peca_atual()
    return recompensa

  def espaco(self):
    self.remover_peca_atual()
    recompensa, tab_atual, tab_anterior = self.tabuleiro.ate_o_final()
    self.desenhar_peca_atual()
    recompensa_reiniciou = self.tabuleiro.reiniciou(tab_anterior, tab_atual)
    self.desenhar_peca_atual()
    if recompensa_reiniciou == punicao_perdeu:
      self.perdeu = True
      self.redesenhar_tudo()
      return punicao_perdeu
    
    self.redesenhar_tudo()

    # punish = compute_punishment(tab_anterior, tab_atual)
    # recompensa_final = punish if punish == -1 else (recompensa + recompensa_reiniciou)
    recompensa_final = recompensa_reiniciou + recompensa
    self.tabuleiro.pontos += recompensa_final
    if self.engine:
      self.engine.desenhar_ponto()
    return recompensa_final

  def s(self):
    self.remover_peca_atual()
    recompensa, tab_atual, tab_anterior = self.tabuleiro.mover_para_baixo()
    if recompensa == nenhuma_recompensa:
      self.desenhar_peca_atual()
      return recompensa

    recompensa = int(recompensa)
    self.desenhar_peca_atual()
    recompensa_reiniciou = self.tabuleiro.reiniciou(tab_anterior, tab_atual)
    self.desenhar_peca_atual()
    if recompensa_reiniciou == punicao_perdeu:
      self.perdeu = True
      self.redesenhar_tudo()
      return punicao_perdeu
    elif recompensa_reiniciou > 0:
      self.redesenhar_tudo()

    # punish = compute_punishment(tab_anterior, tab_atual)
    # recompensa_final = punish if punish == -1 else (recompensa + recompensa_reiniciou)
    recompensa_final = recompensa_reiniciou + recompensa
    self.tabuleiro.pontos += recompensa_final
    if self.engine:
      self.engine.desenhar_ponto()
    return recompensa_final


  def remover_peca_atual(self):
    if self.engine:
      coordenadas = self.tabuleiro.coordenadas()
      for coordenada in coordenadas:
        self.engine.remover_bloco(coordenada)
      self.engine.update()

  def desenhar_peca_atual(self):
    if self.engine:
      coordenadas = self.tabuleiro.coordenadas()
      for coordenada in coordenadas:
        self.engine.desenhar_bloco(coordenada, self.tabuleiro.peca_atual.id)
      self.engine.desenhar_ponto()
      self.engine.update()

  def redesenhar_tudo(self):
    if self.engine:
      for l in range(self.altura_blocos):
        for c in range(self.base_blocos):
          self.engine.desenhar_bloco([l, c], self.tabuleiro.tabuleiro[l+self.tabuleiro.espaco_adicional_cima][c+self.tabuleiro.tamanho_borda])
      self.engine.desenhar_ponto()
      self.engine.update()