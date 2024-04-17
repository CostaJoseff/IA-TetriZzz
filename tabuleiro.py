import sys

from operacoes import matriz_zero_com_bordas, media
from valores import *
import random as rd
import numpy as np
from pecas import Pecas


class Tabuleiro:

  def __init__(self, largura, altura):
    self.espaco_adicional_cima = 5
    self.tamanho_borda = 4
    self.altura_limite = 5

    self.largura = largura
    self.altura = altura

    self.tabuleiro = matriz_zero_com_bordas(altura + self.espaco_adicional_cima, largura, self.tamanho_borda,valor_borda)
    self.tabuleiro = np.array(self.tabuleiro)
    #self.vetor = self.tabuleiro.flatten()

    self.peca_atual = Pecas()
    self.coluna_atual = self.tamanho_borda + 4
    self.linha_atual = self.espaco_adicional_cima
    self.ultima_coluna = self.tamanho_borda + self.largura
    self.primeira_linha = self.altura + self.espaco_adicional_cima
    self.rotacoes = 0

    self.posicionar_peca()

    self.maiores_alturas = [0] * largura
    self.pontos = 0
    self.buracos = 0

  def mover_para_baixo(self):
    self.remover_peca()
    self.linha_atual += 1
    retorno = self.posicionar_peca()
    if retorno == codigo_colidiu:
      self.linha_atual -= 1
      status = self.status()
      self.posicionar_peca()
      return recompensa_minima, status
    self.pontos += 1
    return nenhuma_recompensa, None

  def mover_para_esquerda(self):
    self.remover_peca()
    self.coluna_atual -= 1
    retorno = self.posicionar_peca()
    if retorno == codigo_colidiu:
      self.coluna_atual += 1
      self.posicionar_peca()
      return punicao_leve
    return nenhuma_recompensa

  def mover_para_direita(self):
    self.remover_peca()
    self.coluna_atual += 1
    retorno = self.posicionar_peca()
    if retorno == codigo_colidiu:
      self.coluna_atual -= 1
      self.posicionar_peca()
      return punicao_leve
    return nenhuma_recompensa

  def rotacionar_peca(self):
    punicao_rotacao = 0
    self.remover_peca()
    peca_temp = self.peca_atual.peca_atual
    self.peca_atual.rotacionar()
    if self.peca_atual.rotacoes > 4:
      punicao_rotacao = punicao_leve
    retorno = self.posicionar_peca()
    if retorno == codigo_colidiu:
      self.peca_atual.peca_atual = peca_temp
      self.posicionar_peca()
      return punicao_leve + punicao_rotacao
    return nenhuma_recompensa + punicao_rotacao

  def ate_o_final(self):
    iteracoes = 1
    self.remover_peca()
    status = self.status_atual()
    self.linha_atual += 1
    while self.tem_espaco():
      iteracoes += 1
      self.linha_atual += 1
      self.pontos += 5
    self.linha_atual -= 1
    self.posicionar_peca()
    return iteracoes + recompensa_media, status

  def remover_peca(self):
    for l in range(self.peca_atual.altura() - 1, -1, -1):
      for c in range(self.peca_atual.largura()):
        if self.peca_atual.peca_atual[l][c] != 0:
          self.tabuleiro[self.linha_atual - (self.peca_atual.altura() - 1 - l)][self.coluna_atual + c] = 0

  def posicionar_peca(self):
    if not self.tem_espaco():
      return codigo_colidiu

    for l in range(self.peca_atual.altura() - 1, -1, -1):
      for c in range(self.peca_atual.largura()):
        if self.peca_atual.peca_atual[l][c] != 0:
          self.tabuleiro[self.linha_atual - (self.peca_atual.altura() - 1 - l)][self.coluna_atual + c] = self.peca_atual.peca_atual[l][c]

  def tem_espaco(self):
    for l in range(self.peca_atual.altura() - 1, -1, -1):
      for c in range(self.peca_atual.largura()):
        ocupado = self.peca_atual.peca_atual[l][c] != 0 and self.tabuleiro[self.linha_atual-(self.peca_atual.altura()-1-l)][self.coluna_atual+c] != 0
        if ocupado:
          return False
    return True

  def calcular_alturas(self):
    alturas_peca = self.peca_atual.alturas()
    linha_atual = self.altura - self.linha_atual
    for c in range(self.tamanho_borda, self.tamanho_borda+self.largura):
      for l in range(self.espaco_adicional_cima ,self.primeira_linha):
        if self.tabuleiro[l][c] != 0:
          self.maiores_alturas[c-self.tamanho_borda] = self.altura + self.espaco_adicional_cima - l
          break
        elif l == self.primeira_linha-1:
          self.maiores_alturas[c-self.tamanho_borda] = 0

    return self.maiores_alturas

  def coordenadas(self):
    coordenadas = self.peca_atual.coordenadas()
    for i in range(len(coordenadas)):
      coordenadas[i][0] = self.linha_atual + coordenadas[i][0] - self.espaco_adicional_cima
      coordenadas[i][1] = self.coluna_atual + coordenadas[i][1] - self.tamanho_borda
    return coordenadas

  def vetor2(self):
    return [self.linha_atual, self.coluna_atual]

  def reiniciou(self):
    if self.linha_atual - self.peca_atual.maior_bloco() - 1 <= self.altura_limite: return self.perdeu()
    recompensa_verificacoes = self.verificacoes()
    self.peca_atual.nova_peca()
    self.coluna_atual = self.coluna_atual = self.tamanho_borda + 4
    self.linha_atual = self.espaco_adicional_cima + 1
    self.posicionar_peca()
    return recompensa_verificacoes

  def perdeu(self):
    self.pontos = 0
    self.tabuleiro = matriz_zero_com_bordas(self.altura + self.espaco_adicional_cima, self.largura, self.tamanho_borda,valor_borda)
    self.tabuleiro = np.array(self.tabuleiro)
    self.peca_atual.nova_peca()
    self.coluna_atual = self.coluna_atual = self.tamanho_borda + 4
    self.linha_atual = self.espaco_adicional_cima + 1
    self.posicionar_peca()
    self.maiores_alturas = [0] * self.largura
    self.buracos = 0
    linhas_completas = 0
    return punicao_perdeu

  def calcular_menor_altura(self):
    return min(self.maiores_alturas)
  def calcular_maior_altura(self):
    return max(self.maiores_alturas)

  def verificacoes(self):
    return self.verificar_linhas()

  def verificar_linhas(self, l=None, iteracoes=0):
    if l == None:
      l = len(self.tabuleiro) - self.tamanho_borda - 1

    for l in range(l, 0, -1):
      contem_0 = False
      for c in range(self.tamanho_borda, self.ultima_coluna):
        if self.tabuleiro[l][c] == 0:
          contem_0 = True
          break
      if not contem_0:
        self.swap_baixo_nengue(l)
        return self.verificar_linhas(l, iteracoes + 1)
    recompensa = recompensa_linha_completa * iteracoes
    self.pontos += recompensa
    return recompensa

  def swap_baixo_nengue(self, linha):
    for l in range(linha, 0, -1):
      self.tabuleiro[l] = self.tabuleiro[l - 1]
    for i in range(self.largura):
      self.maiores_alturas[i] -= 1

  def status(self):
    self.calcular_alturas()
    pontos = self.pontos
    maior_altura = self.calcular_maior_altura()
    menor_altura = self.calcular_menor_altura()

    return {"pontos": pontos,
            "maior_altura": maior_altura,
            "menor_altura": menor_altura}

  def status_atual(self):
    pontos = self.pontos
    maior_altura = self.calcular_maior_altura()
    menor_altura = self.calcular_menor_altura()

    return {"pontos": pontos,
            "maior_altura": maior_altura,
            "menor_altura": menor_altura}
