from Jogo.TetriZzz_Otimizado import TetriZzz_Otimizado
from gym.spaces import Discrete, Box
from AI_Hub.FilaDeMovimentos import FilaDeMovimentos
from gym import Env
import numpy as np
import pygame

class TetrizEnv(Env):
    def __init__(self, janela=False):
        self.action_space = Discrete(5)
        self.colunas = 1
        self.linhas = 1
        if janela:
            self.janela = self.criar_janela()
            self.info = pygame.display.Info()
            self.largura_dos_jogos = (self.info.current_w)/self.colunas
            self.altura_dos_jogos = (self.info.current_h)/self.linhas
        else:
            self.janela = False
            self.largura_dos_jogos = 0
            self.altura_dos_jogos = 0
        self.total_de_bots = 1
        self.jogo: TetriZzz_Otimizado = TetriZzz_Otimizado(0*self.largura_dos_jogos, (0+1)*self.largura_dos_jogos, 0*self.altura_dos_jogos, (0+1)*self.altura_dos_jogos, self.janela)
        self.state = self.jogo.tabuleiro.tabuleiro
        self.state_anterior = self.jogo.tabuleiro.tabuleiro
        self.fila_de_movimentos = FilaDeMovimentos()
        self.observation_space = Box(low=0, high=99, shape=np.stack((self.state, self.state_anterior), axis=0).shape, dtype=np.int8)

    def criar_janela(self):
        pygame.init()
        info = pygame.display.Info()
        pygame.display.set_caption("TetriZzz")
        return pygame.display.set_mode([info.current_w/4, info.current_h/2])

    def process_events(self):
        if self.janela:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

    def step(self, action, process_events: bool):
        if process_events:
            self.process_events()
        self.state_anterior = self.jogo.tabuleiro.tabuleiro
        reward = self.jogo.acao(action)
        done = self.jogo.perdeu
        self.state = self.jogo.tabuleiro.tabuleiro
        self.fila_de_movimentos.em_fila(action)
        new_state_return = self.jogo.tabuleiro.model_ajust()
        new_state_return = np.stack((new_state_return, self.state_anterior), axis=0)
        return new_state_return, reward, done, {}

    def reset(self, process_events: bool):
        if process_events:
            self.process_events()
        self.jogo: TetriZzz_Otimizado = TetriZzz_Otimizado(0*self.largura_dos_jogos, (0+1)*self.largura_dos_jogos, 0*self.altura_dos_jogos, (0+1)*self.altura_dos_jogos, self.janela)
        self.state = self.jogo.tabuleiro.tabuleiro
        return np.stack((self.jogo.tabuleiro.model_ajust(), self.state_anterior), axis=0)
