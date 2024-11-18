import random as rd

import pygame, time
from valores import linhas_completas

class Engine:
  def __init__(self, x_i, x_f, y_i, y_f, base_blocos, altura_blocos, janela):
    self.x_i = x_i
    self.x_f = x_f
    self.y_i = y_i
    self.y_f = y_f
    self.largura = x_f - x_i
    self.altura = y_f - y_i
    self.base_blocos = base_blocos
    self.altura_blocos = altura_blocos
    self.largura_do_bloco = self.largura/self.base_blocos
    self.altura_do_bloco = self.altura/self.altura_blocos
    self.janela = janela
    self.fonte_texto = pygame.font.SysFont("Pixeled", 30)
    self.random_background = rd.randint(0, 100)
    janela.fill([self.random_background, self.random_background, self.random_background])
    pygame.display.update()
  
  def fim_de_jogo(self):
    texto = "Aperte Enter\npara reiniciar"
    txt_fim_jogo = self.fonte_texto.render(texto, 1, (255,255,255), (25,25,25))
    self.janela.blit(txt_fim_jogo, (self.largura/3, (self.altura*50)/100))
    pygame.display.update()
    time.sleep(5)

  def desenhar_bloco(self, vetor2, peca):
    if self.y_i+(self.altura_do_bloco*vetor2[0]) < self.y_i: return

    block = pygame.Rect(self.x_i+(self.largura_do_bloco*vetor2[1]), self.y_i+(self.altura_do_bloco*vetor2[0]), self.largura_do_bloco, self.altura_do_bloco)
    match peca:
      case 1: pygame.draw.rect(self.janela, [0,255,10],   block)
      case 2: pygame.draw.rect(self.janela, [255,0,0],    block)
      case 3: pygame.draw.rect(self.janela, [255,0,255],  block)
      case 4: pygame.draw.rect(self.janela, [25,255,255], block)
      case 5: pygame.draw.rect(self.janela, [255,255,0],  block)
      case 6: pygame.draw.rect(self.janela, [255,100,0],  block)
      case 7: pygame.draw.rect(self.janela, [100,50,255], block)
      case 8: pygame.draw.rect(self.janela, [0, 50, 0], block)
      case _: pygame.draw.rect(self.janela, [self.random_background, self.random_background, self.random_background], block)
  
  def update(self):
    pygame.display.update()

  def limpar(self):
    block = pygame.Rect(self.x_i, self.y_i, self.x_f, self.y_f)
    pygame.draw.rect(self.janela, [self.random_background, self.random_background, self.random_background], block)
    pygame.display.update()

  def remover_bloco(self, vetor2):
    if self.y_i + (self.altura_do_bloco * vetor2[0]) < self.y_i: return

    block = pygame.Rect(self.x_i+(self.largura_do_bloco*vetor2[1]), self.y_i+(self.altura_do_bloco*vetor2[0]), self.largura_do_bloco, self.altura_do_bloco)
    pygame.draw.rect(self.janela, [self.random_background, self.random_background, self.random_background], block)
    pygame.display.update()

  def desenhar_ponto(self):
    txt_pontos = self.fonte_texto.render(str(linhas_completas), 0, (255, 255, 255))
    self.janela.blit(txt_pontos, (self.x_i+(self.largura/2), (self.y_i)))