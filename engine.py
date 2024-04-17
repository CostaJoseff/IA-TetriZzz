import random as rd

import pygame, time
from valores import linhas_completas

class Engine:
  def __init__(self, largura, altura, base_blocos, altura_blocos, janela):
    self.largura_tela = 100
    self.largura = largura-self.largura_tela
    self.altura = altura
    self.base_blocos = base_blocos
    self.altura_blocos = altura_blocos
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
    block = pygame.Rect(((self.largura_tela/self.base_blocos)*vetor2[1])+self.largura, (self.altura/self.altura_blocos)*vetor2[0], (self.largura_tela/self.base_blocos), (self.altura/self.altura_blocos))
    match peca:
      case 1: pygame.draw.rect(self.janela, [0,255,10],   block)
      case 2: pygame.draw.rect(self.janela, [255,0,0],    block)
      case 3: pygame.draw.rect(self.janela, [255,0,255],  block)
      case 4: pygame.draw.rect(self.janela, [25,255,255], block)
      case 5: pygame.draw.rect(self.janela, [255,255,0],  block)
      case 6: pygame.draw.rect(self.janela, [255,100,0],  block)
      case 7: pygame.draw.rect(self.janela, [100,50,255], block)
      case _: pygame.draw.rect(self.janela, [self.random_background, self.random_background, self.random_background], block)
  
  def update(self):
    pygame.display.update()

  def limpar(self):
    block = pygame.Rect(((self.largura_tela / self.base_blocos)) + self.largura, (self.altura / self.altura_blocos), self.largura_tela, self.altura)
    pygame.draw.rect(self.janela, [self.random_background, self.random_background, self.random_background], block)
    pygame.display.update()

  def remover_bloco(self, vetor2):
    block = pygame.Rect(((self.largura_tela / self.base_blocos) * vetor2[1]) + self.largura, (self.altura / self.altura_blocos) * vetor2[0], (self.largura_tela / self.base_blocos), (self.altura / self.altura_blocos))
    pygame.draw.rect(self.janela, [self.random_background, self.random_background, self.random_background], block)
    pygame.display.update()

  def desenhar_ponto(self):
    txt_pontos = self.fonte_texto.render(str(linhas_completas), 0, (255, 255, 255))
    self.janela.blit(txt_pontos, (self.largura+(self.largura_tela/2), (self.altura * 1) / 100))