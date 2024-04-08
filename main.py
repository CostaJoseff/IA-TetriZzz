import time
from random import randint
from time import sleep
from tetriZzz_Otimizado import TetriZzz_Otimizado
from operacoes import rotacionar, matriz_to_string

jogo = TetriZzz_Otimizado()

a = 0
while a != -1:
  if a % 140 == 0:
    pass
  #jogo.s()
  jogo.espaco()
  #jogo.w()
  #time.sleep(5)
  a += 1