import sys

from operacoes import matriz_zero_com_bordas
import random as rd
from pecas import Pecas

class Tabuleiro:

    def __init__(self, largura, altura):
        self.x = largura
        self.y = altura
        self.area_extra = 5
        self.tamanho_borda = 4
        self.altura_limite = 4
        self.largura = largura
        self.altura = altura
        self.tabuleiro = matriz_zero_com_bordas(altura+self.area_extra, largura, self.tamanho_borda, -1)
        self.peca_atual = Pecas()
        self.coluna_atual = rd.randint(self.tamanho_borda, largura-3+self.tamanho_borda)
        self.linha_atual = self.area_extra
        self.posicionar_peca()
        self.pontos = 0

        self.maiores_alturas = []
        self.espacos_vazios = 0

    def mover_para_baixo(self):
        self.remover_peca()
        self.linha_atual+=1
        retorno = self.posicionar_peca()
        if retorno == "N/A":
            self.pontos -= 2
            self.linha_atual-=1
            self.posicionar_peca()
            return 0
        self.pontos += 1

    def mover_para_esquerda(self):
        self.remover_peca()
        self.coluna_atual -= 1
        retorno = self.posicionar_peca()
        if retorno == "N/A":
            self.pontos -= 1
            self.coluna_atual += 1
            self.posicionar_peca()
            return -1
        self.pontos += 1

    def mover_para_direita(self):
        self.remover_peca()
        self.coluna_atual += 1
        retorno = self.posicionar_peca()
        if retorno == "N/A":
            self.pontos -= 1
            self.coluna_atual -= 1
            self.posicionar_peca()
            return -1
        self.pontos += 1


    def rotacionar_peca(self):
        self.remover_peca()
        peca_temp = self.peca_atual.peca_atual
        self.peca_atual.rotacionar()
        retorno = self.posicionar_peca()
        if retorno == "N/A":
            self.pontos -= 1
            self.peca_atual.peca_atual = peca_temp
            self.posicionar_peca()
            return -1
        self.pontos += 1

    def ate_o_final(self):
        self.remover_peca()
        self.linha_atual += 1
        while self.tem_espaco():
            self.linha_atual += 1
            self.pontos += 2
        self.linha_atual -= 1
        self.posicionar_peca()
        self.verificacoes()
        return 0


    def remover_peca(self):
        for l in range(self.peca_atual.altura() - 1, -1, -1):
            for c in range(self.peca_atual.largura()):
                if self.peca_atual.peca_atual[l][c] != 0:
                    self.tabuleiro[self.linha_atual - (self.peca_atual.altura() - 1 - l)][self.coluna_atual + c] = 0

    def posicionar_peca(self):
        if not self.tem_espaco(): return "N/A"

        for l in range(self.peca_atual.altura()-1, -1, -1):
            for c in range(self.peca_atual.largura()):
                if self.peca_atual.peca_atual[l][c] != 0:
                    self.tabuleiro[self.linha_atual-(self.peca_atual.altura()-1-l)][self.coluna_atual+c] = self.peca_atual.peca_atual[l][c]

    def tem_espaco(self):
        for l in range(self.peca_atual.altura()-1, -1, -1):
            for c in range(self.peca_atual.largura()):
                ocupado = self.peca_atual.peca_atual[l][c] != 0 and self.tabuleiro[self.linha_atual-(self.peca_atual.altura()-1-l)][self.coluna_atual+c] != 0
                if ocupado:
                    return False
        return True

    def status(self):
        status = self.calcular_alturas()
        status.append(self.coluna_atual)
        status.append(self.linha_atual)
        status.append(self.peca_atual.id)
        status.append(self.peca_atual.rotacao)
        return status

    def calcular_alturas(self):
        alturas = []
        for c in range(self.tamanho_borda, self.tamanho_borda+self.largura):
            for l in range(0, self.altura+self.area_extra):
                if self.tabuleiro[l][c] != 0:
                    alturas.append(l)
                    break
        self.maiores_alturas = alturas

    def coordenadas(self):
        coordenadas = self.peca_atual.coordenadas()
        for i in range(len(coordenadas)):
            coordenadas[i][0] = self.linha_atual + coordenadas[i][0] - self.area_extra
            coordenadas[i][1] = self.coluna_atual + coordenadas[i][1] - self.tamanho_borda
        return coordenadas

    def vetor2(self):
        return [self.linha_atual, self.coluna_atual]

    def reiniciou(self):
        if self.linha_atual-self.peca_atual.maior_bloco()-1 <= self.altura_limite: return self.perdeu()
        pontuacao_anterior = self.pontos
        self.verificacoes()

        self.peca_atual.nova_peca()
        self.coluna_atual = rd.randint(self.tamanho_borda, self.largura-3+self.tamanho_borda)
        self.linha_atual = self.area_extra+1
        self.posicionar_peca()
        return self.pontos - pontuacao_anterior

    def perdeu(self):
        self.pontos = 0
        self.tabuleiro = matriz_zero_com_bordas(self.y + self.area_extra, self.x, self.tamanho_borda, -1)
        self.peca_atual.nova_peca()
        self.coluna_atual = rd.randint(self.tamanho_borda, self.largura - 3 + self.tamanho_borda)
        self.linha_atual = self.area_extra+1
        self.posicionar_peca()
        self.maiores_alturas = []
        self.espacos_vazios = 0
        return -5

    def contar_espacos_vazios(self):
        resultado = 0
        for c in range(len(self.maiores_alturas)):
            for l in range(len(self.tabuleiro)-self.tamanho_borda, self.maiores_alturas[c]):
                if self.tabuleiro[l][c] == 0 and l < self.maiores_alturas[c]:
                    resultado += 1

        self.maiores_alturas = resultado

    def verificacoes(self):
        self.verificar_linhas()
        return self.pontos

    def verificar_linhas(self, l = None, iteracoes = 0):
        if l == None:
            l = len(self.tabuleiro)-self.tamanho_borda-1

        for l in range(l, 0, -1):
            contem_0 = False
            for c in range(self.tamanho_borda, self.tamanho_borda+self.largura):
                if self.tabuleiro[l][c] == 0:
                    contem_0 = True
                    break
            if not contem_0:
                self.swap_baixo_nengue(l)
                return self.verificar_linhas(l, iteracoes+1)
        self.pontos += 3*iteracoes

    def swap_baixo_nengue(self, linha):
        for l in range(linha, 0, -1):
            self.tabuleiro[l] = self.tabuleiro[l-1]