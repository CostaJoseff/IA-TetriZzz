from Jogo.valores import linhas_completas
from Jogo.Tabuleiro import Tabuleiro
import pygame, time

class Engine:
  def __init__(self, x_i, x_f, y_i, y_f, base_blocos, altura_blocos, janela, tabuleiro: Tabuleiro=None):
    self.x_i = x_i
    self.x_f = x_f
    self.y_i = y_i
    self.y_f = y_f
    self.tabuleiro = tabuleiro
    self.largura = x_f - x_i
    self.altura = y_f - y_i
    self.base_blocos = base_blocos
    self.altura_blocos = altura_blocos
    self.largura_do_bloco = self.largura/self.base_blocos
    self.altura_do_bloco = self.altura/self.altura_blocos
    self.janela = janela
    self.fonte_texto = pygame.font.SysFont("Pixeled", 30)
    self.random_background = 50# rd.randint(0, 100)
    janela.fill([self.random_background, self.random_background, self.random_background])
    pygame.display.update()
  
  def fim_de_jogo(self):
    if self.janela:
      texto = "Aperte Enter\npara reiniciar"
      txt_fim_jogo = self.fonte_texto.render(texto, 1, (255,255,255), (25,25,25))
      self.janela.blit(txt_fim_jogo, (self.largura/3, (self.altura*50)/100))
      pygame.display.update()
      time.sleep(5)

  def desenhar_bloco(self, vetor2, peca):
    if not self.janela:
       return
    if self.y_i+(self.altura_do_bloco*vetor2[0]) < self.y_i: return

    block = pygame.Rect(self.x_i+(self.largura_do_bloco*vetor2[1]), self.y_i+(self.altura_do_bloco*vetor2[0]), self.largura_do_bloco, self.altura_do_bloco)
    if peca == 1:
        pygame.draw.rect(self.janela, [0, 255, 10], block)
    elif peca == 2:
        pygame.draw.rect(self.janela, [255, 0, 0], block)
    elif peca == 3:
        pygame.draw.rect(self.janela, [255, 0, 255], block)
    elif peca == 4:
        pygame.draw.rect(self.janela, [25, 255, 255], block)
    elif peca == 5:
        pygame.draw.rect(self.janela, [255, 255, 0], block)
    elif peca == 6:
        pygame.draw.rect(self.janela, [255, 100, 0], block)
    elif peca == 7:
        pygame.draw.rect(self.janela, [100, 50, 255], block)
    elif peca == 8:
        pygame.draw.rect(self.janela, [0, 250, 250], block)
    elif peca == 99:
       pygame.draw.rect(self.janela, [200, 200, 200], block)
    else:
        pygame.draw.rect(self.janela, [self.random_background, self.random_background, self.random_background], block)

  
  def update(self):
    if self.janela:
      pygame.display.update()

  def limpar(self):
    if self.janela:
      block = pygame.Rect(self.x_i, self.y_i, self.x_f, self.y_f)
      pygame.draw.rect(self.janela, [self.random_background, self.random_background, self.random_background], block)
      pygame.display.update()

  def remover_bloco(self, vetor2):
    if self.y_i + (self.altura_do_bloco * vetor2[0]) < self.y_i: return

    if self.janela:
      block = pygame.Rect(self.x_i+(self.largura_do_bloco*vetor2[1]), self.y_i+(self.altura_do_bloco*vetor2[0]), self.largura_do_bloco, self.altura_do_bloco)
      pygame.draw.rect(self.janela, [self.random_background, self.random_background, self.random_background], block)
      # pygame.display.update()

  def desenhar_ponto(self):
    if self.janela:
      txt_pontos = self.fonte_texto.render(str(int(self.tabuleiro.pontos) if self.tabuleiro.pontos != -float("inf") else "-inf"), 0, (255, 255, 255))
      self.janela.blit(txt_pontos, (self.x_i+(self.largura/2), (self.y_i)))