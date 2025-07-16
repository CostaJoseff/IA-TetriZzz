from Jogo2.valores import punicao_perdeu, nenhuma_recompensa
from Jogo2.Tabuleiro import Tabuleiro
from gym.spaces import Discrete, Box
from copy import deepcopy as copy
from Jogo2.Engine import Engine
import numpy as np
import pygame

class TetriZzz:
    def __init__(self, janela=False):
        self.colunas = 1
        self.linhas = 1
        self.altura_blocos = 16
        self.base_blocos = 10
        self.tabuleiro: Tabuleiro = Tabuleiro(self.base_blocos, self.altura_blocos)
        self.estado_atual = copy(self.tabuleiro.tabuleiro)
        self.estado_anterior = copy(self.estado_atual)
        self.observation_space = Box(low=0, high=99, shape=self.estado_atual.shape, dtype=np.int8)
        self.action_space = Discrete(5)
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.perdeu = False
        if janela:
            self.janela = self.criar_janela()
            self.info = pygame.display.Info()
            self.largura_dos_jogos = (self.info.current_w)/self.colunas
            self.altura_dos_jogos = (self.info.current_h)/self.linhas
            self.engine: Engine = Engine(0, self.largura_dos_jogos, 0, self.altura_dos_jogos, self.base_blocos, self.altura_blocos, self.janela, self.tabuleiro)
        else:
            self.janela = None
            self.engine = None

    def process_events(self):
        if self.janela:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

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

    def a(self):
        recompensa = self.tabuleiro.mover_para_esquerda()
        self.tabuleiro.pontos += recompensa
        self.desenhar_peca_atual()
        return recompensa
    
    def w(self):
        recompensa = self.tabuleiro.rotacionar_peca()
        self.tabuleiro.pontos += recompensa
        self.desenhar_peca_atual()
        return recompensa
    
    def d(self):
        recompensa = self.tabuleiro.mover_para_direita()
        self.tabuleiro.pontos += recompensa
        self.desenhar_peca_atual()
        return recompensa
    
    def espaco(self):
        recompensa = self.tabuleiro.ate_o_final()
        self.desenhar_peca_atual()
        recompensa_reiniciou = self.tabuleiro.reiniciou(self.estado_anterior, self.tabuleiro.tabuleiro)
        self.desenhar_peca_atual()
        if recompensa_reiniciou == punicao_perdeu:
            self.perdeu = True
            self.redesenhar_tudo()
            return punicao_perdeu
        
        self.redesenhar_tudo()

        recompensa_final = recompensa_reiniciou
        self.tabuleiro.pontos += recompensa_final
        if self.engine:
            self.engine.desenhar_ponto()
        return recompensa_final
    
    def s(self):
        recompensa = self.tabuleiro.mover_para_baixo()
        if recompensa == nenhuma_recompensa:
            self.desenhar_peca_atual()
            return recompensa

        recompensa = int(recompensa)
        self.desenhar_peca_atual()
        recompensa_reiniciou = self.tabuleiro.reiniciou(self.estado_anterior, self.tabuleiro.tabuleiro)
        self.desenhar_peca_atual()
        if recompensa_reiniciou == punicao_perdeu:
            self.perdeu = True
            self.redesenhar_tudo()
            return punicao_perdeu
        elif recompensa_reiniciou != 0:
            self.redesenhar_tudo()

        recompensa_final = recompensa_reiniciou
        self.tabuleiro.pontos += recompensa_final
        if self.engine:
            self.engine.desenhar_ponto()
        return recompensa_final

    def step(self, acao, process_events: bool = True):
        if process_events:
            self.process_events()
        self.clock.tick(self.fps)
        self.remover_peca_atual()
        self.tabuleiro.remover_peca()
        self.estado_anterior = copy(self.tabuleiro.tabuleiro)
        retr = None
        if acao == 0:
            retr = self.w()
        elif acao == 1:
            retr = self.a()
        elif acao == 2:
            retr = self.s()
        elif acao == 3:
            retr = self.d()
        elif acao == 4:
            retr = self.espaco()
        self.estado_atual = copy(self.tabuleiro.tabuleiro)
        return self.estado_atual, retr, self.perdeu, {}

    def reset(self, process_events: bool = True):
        if process_events:
            self.process_events()
        self.tabuleiro: Tabuleiro = Tabuleiro(self.base_blocos, self.altura_blocos)
        self.redesenhar_tudo()
        self.estado_atual = copy(self.tabuleiro.tabuleiro)
        self.estado_anterior = copy(self.estado_atual)
        self.perdeu = False
        self.redesenhar_tudo()
        return self.estado_atual

    def criar_janela(self):
        # pygame.quit()
        pygame.init()
        info = pygame.display.Info()
        pygame.display.set_caption("TetriZzz")
        return pygame.display.set_mode([info.current_w/4, info.current_h/2])