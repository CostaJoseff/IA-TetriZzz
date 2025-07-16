from Util.Operacoes import matriz_zero_com_bordas
from copy import deepcopy as copy
from threading import Thread
from Jogo2.Pecas import Pecas
from Jogo2.valores import *
import numpy as np

class Tabuleiro:

    def __init__(self, largura, altura):
        self.espaco_adicional_cima = 5
        self.tamanho_borda = tamanho_borda
        self.altura_limite = 5

        self.largura = largura
        self.altura = altura

        self.tabuleiro: np.ndarray = matriz_zero_com_bordas(altura + self.espaco_adicional_cima, largura, self.tamanho_borda, valor_borda)
        self.peca_atual = Pecas()
        self.coluna_atual = self.tamanho_borda + 4
        self.linha_atual = self.espaco_adicional_cima
        self.ultima_coluna = self.tamanho_borda + self.largura
        self.primeira_linha = self.altura + self.espaco_adicional_cima
        self.rotacoes = 0
        self.pontos = 0
        self.bloqueados = 0
        self.maiores_alturas = [0] * largura

    def remover_peca(self):
        for l in range(self.peca_atual.altura() - 1, -1, -1):
            for c in range(self.peca_atual.largura()):
                if self.peca_atual.peca_atual[l][c] != 0:
                    self.tabuleiro[self.linha_atual - (self.peca_atual.altura() - 1 - l)][self.coluna_atual + c] = 0

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
    
    def model_ajust(self, input: np.ndarray = None):
        if input is None:
            tab = self.tabuleiro.copy()
        else:
            tab = input.copy()

        for l in range(len(tab)):
            for c in range(len(tab[0])):
                tab[l][c] = 1 if tab[l][c] > 0 and tab[l][c] != self.peca_atual.id else tab[l][c]

        tab: np.ndarray = np.array(tab)
        if input is None:
            self.tabuleiro = tab
        return tab
    
    def mover_para_baixo(self):
        self.remover_peca()
        self.linha_atual += 1
        retorno = self.posicionar_peca()
        if retorno == codigo_colidiu:
            self.linha_atual -= 1
            self.posicionar_peca()
            self.model_ajust()
            return 0.1
        return nenhuma_recompensa

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
            punicao_rotacao = 0
        retorno = self.posicionar_peca()
        if retorno == codigo_colidiu:
            self.peca_atual.peca_atual = peca_temp
            self.posicionar_peca()
            return punicao_leve + punicao_rotacao
        return nenhuma_recompensa + punicao_rotacao

    def ate_o_final(self):
        self.remover_peca()
        self.linha_atual += 1
        while self.tem_espaco():
            self.linha_atual += 1
            #self.pontos += 5
        self.linha_atual -= 1
        self.posicionar_peca()
        # self.pontos += recompensa_media
        self.model_ajust()
        return recompensa_media
    
    def calcular_alturas(self, tab=None):
        if tab is None:
            tab = self.tabuleiro

        maiores_alturas = [0] * self.largura
        for c in range(self.tamanho_borda, self.tamanho_borda+self.largura):
            for l in range(self.espaco_adicional_cima ,self.primeira_linha):
                if tab[l][c] != 0:
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
    
    def chegar_no_topo(self, l, c, tabul, dir_anterior=0):
        em_volta_algum_chega = (isinstance(tabul[l-1][c], bool) and tabul[l-1][c] == True) or (isinstance(tabul[l][c-1], bool) and tabul[l][c-1] == True) or (isinstance(tabul[l][c+1], bool) and tabul[l][c+1] == True)
        if em_volta_algum_chega:
            tabul[l][c] = True
            return True
        
        em_volta_nenhum_chega = (isinstance(tabul[l-1][c], bool) and tabul[l-1][c] == False) and (isinstance(tabul[l][c-1], bool) and tabul[l][c-1] == False) and (isinstance(tabul[l][c+1], bool) and tabul[l][c+1] == False)
        if em_volta_nenhum_chega:
            tabul[l][c] = False
            return False

        sem_caminho = tabul[l-1][c] != 0 and tabul[l][c+1] != 0 and tabul[l][c-1] != 0
        if sem_caminho:
            tabul[l][c] = False
            return False
        
        ultima_linha = 0 == l-1
        if ultima_linha:
            tabul[l][c] = True
            return True
        
        cima, esquerda, direita = False, False, False

        if isinstance(tabul[l-1][c], bool):
            cima = tabul[l-1][c]
        elif tabul[l-1][c] == 0:
            cima = self.chegar_no_topo(l-1, c, tabul)

        if isinstance(tabul[l][c-1], bool):
            esquerda = tabul[l][c-1]
        elif tabul[l][c-1] == 0 and dir_anterior != 1 and not cima:
            esquerda = self.chegar_no_topo(l, c-1, tabul, -1)

        if isinstance(tabul[l][c+1], bool):
            direita = tabul[l][c+1]
        elif tabul[l][c+1] == 0 and dir_anterior != -1 and not esquerda and not cima:
            direita = self.chegar_no_topo(l, c+1, tabul, 1)
        
        tabul[l][c] = cima or esquerda or direita
        return cima or esquerda or direita
    
    def novos_espaços_trancados(self):
        global novos_espacos_trancados
        l_atual = 0
        c_atual = 0
        bloqueados = 0

        c_atual = self.tamanho_borda
        l_atual = self.altura+self.espaco_adicional_cima-1

        tab_copy = np.array(copy(self.tabuleiro), dtype=object)

        for c in range(self.largura):
            for l in range(self.altura):
                if self.tabuleiro[l_atual-l][c_atual+c] > 0:
                    continue
                if self.tabuleiro[l_atual-l][c_atual+c] == -1:
                    break

                if not self.chegar_no_topo(l_atual-l, c_atual+c, tab_copy):
                    bloqueados += 1
                else:
                    continue

        novos_bloqueados = bloqueados - self.bloqueados
        self.bloqueados = bloqueados
        novos_espacos_trancados = novos_bloqueados

    def swap_baixo_nengue(self, linha):
        for l in range(linha, 0, -1):
            self.tabuleiro[l] = self.tabuleiro[l - 1]
        for i in range(self.largura):
            self.maiores_alturas[i] -= 1

    def verificar_linhas_completas(self, l=None, iteracoes=0):
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
                return self.verificar_linhas_completas(l, iteracoes + 1)
        recompensa = recompensa_linha_completa * iteracoes
        return recompensa
    
    def calcular_bumpness_score(self, tab_anterior, tab_atual):
        global bumpness_score
        try:
            alturas_anterior = self.calcular_alturas(tab_anterior)
            alturas_atual = self.calcular_alturas(tab_atual)
            bumpness_anterior_score = sum(abs(alturas_anterior[i] - alturas_anterior[i+1]) for i in range(len(alturas_anterior)-1))
            if bumpness_anterior_score == 0:
                bumpness_score = 0
                return
            bumpness_atual_score = sum(abs(alturas_atual[i] - alturas_atual[i+1]) for i in range(len(alturas_atual)-1))
            delta = bumpness_atual_score - bumpness_anterior_score
            bumpness_score = -delta if delta is not None else 0
        except:
            bumpness_score = 0

    def avaliar_empilhamento(self, altura_anterior, altura_atual):
        global pilha_score
        delta = [a - b for a, b in zip(altura_atual, altura_anterior)]
        posicoes_usadas = [i for i, d in enumerate(delta) if d > 0]
        
        if not posicoes_usadas:
            return 0

        menor_altura = min(altura_anterior)

        maiores_diferencas = [
            altura_atual[i] - menor_altura for i in posicoes_usadas
        ]

        pior_diferenca = max(maiores_diferencas)

        if pior_diferenca < 5:
            pilha_score = 0
        else:
            valor = max(-0.99, - (pior_diferenca - 4) / 6)
            pilha_score = round(valor, 2)
    
    def reiniciou(self, tab_anterior=None, tab_atual=None):
        if self.linha_atual - 2 <= self.altura_limite: return punicao_perdeu

        threads = []

        global novos_espacos_trancados
        novos_espacos_trancados = 0
        thread = Thread(target=self.novos_espaços_trancados)
        thread.start()
        threads.append(thread)

        global recompensa_linhas_completas
        recompensa_linhas_completas = 0
        thread = Thread(target=self.verificar_linhas_completas)
        thread.start()
        threads.append(thread)

        global bumpness_score
        bumpness_score = 0
        thread = Thread(target=self.calcular_bumpness_score, args=[tab_anterior, tab_atual])
        thread.start()
        threads.append(thread)

        global pilha_score
        pilha_score = 0
        thread = Thread(target=self.avaliar_empilhamento, args=[self.calcular_alturas(tab_anterior), self.calcular_alturas(tab_atual)])
        thread.start()
        threads.append(thread)

        thr: Thread
        for thr in threads:
            thr.join()

        self.peca_atual.nova_peca()
        self.coluna_atual = self.tamanho_borda + 4
        self.linha_atual = self.espaco_adicional_cima + 1
        self.posicionar_peca()

        if recompensa_linhas_completas > 0: # Prioridade é benificiar linhas completas
            return 1 + (0 if novos_espacos_trancados >= 0 else 1)
        elif bumpness_score < -3: # Punir se prejudicar o bumpness independente de qualquer coisa
            return punicao_bumpness
        elif novos_espacos_trancados > 1: # Punir se existir espaço trancado independente de bumpness bom
            return punicao_alta
        elif pilha_score != 0: # Punir se empilhar DEMAIS
            return pilha_score
        elif bumpness_score >= -3: # Se bumpness bom da recompensa
            return recompensa_bumpness
        else: # Se nenhum desses, então 0
            return 0
