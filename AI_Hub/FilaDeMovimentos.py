from AI_Hub.valores import tamanho_fila_movimentos
import numpy as np

class FilaDeMovimentos():
    def __init__(self):
        self.fila = [-1]*tamanho_fila_movimentos
        self.tamanho = tamanho_fila_movimentos

    def em_fila(self, elemento):
        self.fila.append(elemento)
        self.fila.pop(0)

    def to_numpy(self):
        return np.array(self.fila)