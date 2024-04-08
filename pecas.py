import random as rd
from operacoes import rotacionar

id_i = 1
id_j = 2
id_l = 3
id_o = 4
id_s = 5
id_z = 6
id_t = 7


def gerar_pecas():
    peca_i = [[0, 1, 0, 0],
              [0, 1, 0, 0],
              [0, 1, 0, 0],
              [0, 1, 0, 0]]

    peca_j = [[0, 2, 0],
              [0, 2, 0],
              [2, 2, 0]]

    peca_l = [[0, 3, 0],
              [0, 3, 0],
              [0, 3, 3]]

    peca_o = [[4, 4],
              [4, 4]]

    peca_s = [[0, 5, 5],
              [5, 5, 0],
              [0, 0, 0]]

    peca_z = [[6, 6, 0],
              [0, 6, 6],
              [0, 0, 0]]

    peca_t = [[0, 0, 0],
              [7, 7, 7],
              [0, 7, 0]]
    return [peca_i, peca_j, peca_l, peca_o, peca_s, peca_z, peca_t]


class Pecas:

    def __init__(self):
        self.rotacao = rd.randint(0, 3)
        self.pecas = gerar_pecas()
        self.id = rd.randint(1, 7)
        self.peca_atual = self.pecas[self.id - 1]

        for _ in range(self.rotacao):
            self.peca_atual = rotacionar(self.peca_atual)

    def rotacionar(self):
        self.peca_atual = rotacionar(self.peca_atual)
        self.rotacao = (self.rotacao + 1) % 4

    def nova_peca(self):
        self.id = rd.randint(1, 7)
        self.peca_atual = self.pecas[self.id - 1]
        self.rotacao = rd.randint(0, 3)
        for _ in range(self.rotacao):
            self.peca_atual = rotacionar(self.peca_atual)

    def altura(self):
        return len(self.peca_atual)

    def largura(self):
        return len(self.peca_atual[0])

    def coordenadas(self):
        coordenadas = []
        for l in range(self.altura() - 1, -1, -1):
            for c in range(self.largura()):
                if self.peca_atual[l][c] != 0:
                    coordenadas.append([l - self.altura() + 1, c])

        return coordenadas

    def maior_bloco(self):
        for l in range(len(self.peca_atual)):
            for c in range(len(self.peca_atual[0])):
                if self.peca_atual[l][c] != 0: return l