from Util.Operacoes import matriz_zero_com_bordas
from Jogo.Pecas import Pecas
from Jogo.valores import *
import numpy as np


class Tabuleiro:

  def __init__(self, largura, altura):
    self.espaco_adicional_cima = 5
    self.tamanho_borda = tamanho_borda
    self.altura_limite = 5

    self.largura = largura
    self.altura = altura

    self.tabuleiro: np.ndarray = matriz_zero_com_bordas(altura + self.espaco_adicional_cima, largura, self.tamanho_borda, valor_borda)
    # self.tabuleiro = np.squeeze(self.tabuleiro, -1)
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
    tab_anterior = self.tabuleiro.copy()
    self.linha_atual += 1
    retorno = self.posicionar_peca()
    if retorno == codigo_colidiu:
      self.linha_atual -= 1
      self.posicionar_peca()
      tab_atual = self.tabuleiro.copy()
      return 0.0000001, tab_atual, tab_anterior
    return nenhuma_recompensa, None, None

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
    self.remover_peca()
    tab_anterior = self.tabuleiro.copy()
    self.linha_atual += 1
    while self.tem_espaco():
      self.linha_atual += 1
      self.pontos += 5
    self.linha_atual -= 1
    self.posicionar_peca()
    tab_atual = self.tabuleiro.copy()
    
    return recompensa_media, tab_atual, tab_anterior

  def remover_peca(self):
    for l in range(self.peca_atual.altura() - 1, -1, -1):
      for c in range(self.peca_atual.largura()):
        if self.peca_atual.peca_atual[l][c] != 0:
          self.tabuleiro[self.linha_atual - (self.peca_atual.altura() - 1 - l)][self.coluna_atual + c] = 0

  def model_ajust(self, input: np.ndarray = None):
    if not input:
      tab = self.tabuleiro.copy()
    else:
      tab = input.copy()
      
    self.peca_atual.model_ajust()
    for l in range(self.peca_atual.altura() - 1, -1, -1):
      for c in range(self.peca_atual.largura()):
        if self.peca_atual.peca_atual[l][c] != 0:
          if self.peca_atual.peca_atual[l][c] != 99:
            tab[self.linha_atual - (self.peca_atual.altura() - 1 - l)][self.coluna_atual + c] = self.peca_atual.peca_atual[l][c]
          else:
            tab[self.linha_atual - (self.peca_atual.altura() - 1 - l)][self.coluna_atual + c] = self.peca_atual.peca_atual_modelo[l][c]

    tab: np.ndarray = np.array(tab)
    return tab.flatten()

  def posicionar_peca(self):
    if not self.tem_espaco(): return codigo_colidiu

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
    maiores_alturas = [0] * self.largura
    for c in range(self.tamanho_borda, self.tamanho_borda+self.largura):
      for l in range(self.espaco_adicional_cima ,self.primeira_linha):
        if self.tabuleiro[l][c] != 0:
          maiores_alturas[c-self.tamanho_borda] = self.altura + self.espaco_adicional_cima - l
          break
        elif l == self.primeira_linha-1:
          maiores_alturas[c-self.tamanho_borda] = 0

    return maiores_alturas

  def coordenadas(self):
    coordenadas = self.peca_atual.coordenadas()
    for i in range(len(coordenadas)):
      coordenadas[i][0] = self.linha_atual + coordenadas[i][0] - self.espaco_adicional_cima
      coordenadas[i][1] = self.coluna_atual + coordenadas[i][1] - self.tamanho_borda
    return coordenadas

  def vetor2(self):
    return [self.linha_atual, self.coluna_atual]

  def reiniciou(self, tab_anterior=None, tab_atual=None):
    if self.linha_atual - self.peca_atual.maior_bloco() - 1 <= self.altura_limite: return punicao_perdeu

    linha_final = self.espaco_adicional_cima + self.altura - self.linha_atual
    menor_altura = min(self.calcular_alturas())
    dif_linha_final = linha_final - menor_altura
    punicao_altura = -7 if dif_linha_final >= 4 else 0
    
    recompensa_verificacoes = self.verificacoes(tab_anterior, tab_atual)

    self.peca_atual.nova_peca()
    self.coluna_atual = self.tamanho_borda + 4
    self.linha_atual = self.espaco_adicional_cima + 1
    self.posicionar_peca()

    return recompensa_verificacoes + punicao_altura

  def verificacoes(self, tab_anterior=None, tab_atual=None):
    if tab_atual is not None and tab_anterior is not None:
      return self.verificar_linhas_2(tab_anterior, tab_atual) + self.verificar_linhas()
    return self.verificar_linhas()
  
  def verificar_linhas_2(self, tab_anterior=None, tab_atual=None):
    contagem_anterior = (tab_anterior > 0).sum(axis=1)
    contagem_atual = (tab_atual > 0).sum(axis=1)

    mascara = contagem_anterior > 0
    diferenca = contagem_atual[mascara] - contagem_anterior[mascara]

    if np.any(diferenca > 0):
      return 1
    else:
      return 0

  def verificar_linhas(self, l=None, iteracoes=0):
    if l == None:
      l = len(self.tabuleiro) - self.tamanho_borda - 1

    for l in range(l, 3, -1):
      contem_0 = False
      for c in range(self.tamanho_borda, self.ultima_coluna):
        if self.tabuleiro[l][c] == 0:
          contem_0 = True
          break
      if not contem_0:
        self.swap_baixo_nengue(l)
        return self.verificar_linhas(l, iteracoes + 1)
    recompensa = recompensa_linha_completa * iteracoes
    return recompensa

  def swap_baixo_nengue(self, linha):
    for l in range(linha, 0, -1):
      self.tabuleiro[l] = self.tabuleiro[l - 1]
    for i in range(self.largura):
      self.maiores_alturas[i] -= 1
