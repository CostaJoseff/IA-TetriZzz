import random as rd
from operacoes import rotacionar

id_i = 1
id_j = 2
id_l = 3
id_o = 4
id_s = 5
id_z = 6
id_t = 7


def gerar_pecas ():
  peca_i = [[0, 8, 0, 0],
            [0, 8, 0, 0],
            [0, 8, 0, 0],
            [0, 8, 0, 0]]

  peca_j = [[0, 8, 0],
            [0, 8, 0],
            [8, 8, 0]]

  peca_l = [[0, 8, 0],
            [0, 8, 0],
            [0, 8, 8]]

  peca_o = [[8, 8],
            [8, 8]]

  peca_s = [[0, 8, 8],
            [8, 8, 0],
            [0, 0, 0]]

  peca_z = [[8, 8, 0],
            [0, 8, 8],
            [0, 0, 0]]

  peca_t = [[0, 0, 0],
            [8, 8, 8],
            [0, 8, 0]]
  return [peca_i, peca_j, peca_l, peca_o, peca_s, peca_z, peca_t]


class Pecas:

  def __init__ (self):
    self.rotacao = rd.randint(0, 3)
    self.pecas = gerar_pecas()
    self.id = rd.randint(1, 7)
    self.peca_atual = self.pecas[self.id - 1]
    self.rotacoes = 0

    for _ in range(self.rotacao):
      self.peca_atual = rotacionar(self.peca_atual)

  def rotacionar (self):
    self.peca_atual = rotacionar(self.peca_atual)
    self.rotacao = (self.rotacao + 1) % 4
    self.rotacoes += 1

  def nova_peca (self):
    self.id = rd.randint(1, 7)
    self.peca_atual = self.pecas[self.id - 1]
    self.rotacao = rd.randint(0, 3)
    self.rotacoes = 0
    for _ in range(self.rotacao):
      self.peca_atual = rotacionar(self.peca_atual)

  def altura (self):
    return len(self.peca_atual)

  def largura (self):
    return len(self.peca_atual[0])

  def coordenadas (self):
    coordenadas = []
    for l in range(self.altura() - 1, -1, -1):
      for c in range(self.largura()):
        if self.peca_atual[l][c] != 0:
          coordenadas.append([l - self.altura() + 1, c])

    return coordenadas

  def maior_bloco (self):
    for l in range(len(self.peca_atual)):
      for c in range(len(self.peca_atual[0])):
        if self.peca_atual[l][c] != 0: return l

  def alturas (self):
    if self.id == id_i:
      pass

    alturas = []
    for c in range(len(self.peca_atual[0])):
      for l in range(len(self.peca_atual)):
        if self.peca_atual[l][c] != 0:
          alturas.append(self.altura()-l)
          break
        if l == len(self.peca_atual) - 1:
          alturas.append(0)
          break
    return alturas
