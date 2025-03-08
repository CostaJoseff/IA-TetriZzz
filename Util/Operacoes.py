import numpy as np
import time
from Jogo.valores import tamanho_borda

def get_column_heights(board):
    """
    Percorre o tabuleiro coluna a coluna (de cima para baixo) e retorna uma lista
    com a 'altura' da peça mais alta de cada coluna.
    
    A altura é definida como:
       altura = total_de_linhas - índice_da_linha_da_peça
       
    Se nenhuma peça (valor > 0) for encontrada em uma coluna, a altura é 0.
    
    Parâmetros:
      board: matriz NumPy com valores:
             -1  -> borda
              0  -> espaço livre
             >0  -> peça
             
    Retorna:
      Uma lista de tamanho igual ao número de colunas.
    """
    rows, cols = board.shape
    heights = [0] * cols
    for col in range(cols):
        found = False
        for row in range(rows):
            # Ignora borda e espaço livre
            if board[row, col] != -1 and board[row, col] != 0:
                # Calcula a altura: peças mais acima terão valor maior
                heights[col] = rows - row - tamanho_borda
                found = True
                break
        if not found:
            heights[col] = 0
    return np.array(heights)

def coef_var_local(arr):
    diffs = np.abs(np.diff(arr)) 
    mean_val = np.mean(arr) if np.mean(arr) != 0 else 1 
    return np.mean(diffs) / mean_val

def compute_punishment(prev_board, curr_board):
    """
    Compara os tabuleiros anterior e atual e calcula a punição com base em:
      1. Aumento da maior altura (diferença entre os valores máximos)
      2. Aumento da menor altura (entre colunas com peça) em pelo menos 4
      3. Quão "plano" está o conjunto das alturas (usando desvio padrão)
      
    Retorna o valor final da punição.
    """
    prev_heights = get_column_heights(prev_board)
    curr_heights = get_column_heights(curr_board)

    punishs = False

    # 1. Punição para aumento da maior altura
    prev_max = max(prev_heights)
    curr_max = max(curr_heights)
    kurt = coef_var_local(curr_heights)
    # print(f"Kurt {kurt}")
    if (curr_max >= 5 and curr_max > prev_max and curr_max - prev_max >= 2):# and kurt >= 0.4:
        punishs = True

    return -1 if punishs else 0

def rotacionar(matriz):
    resultado = []
    for c in range(len(matriz[0])):
        linha = []
        for l in range(len(matriz)-1, -1, -1):
            linha.append(matriz[l][c])
        resultado.append(linha)
    return resultado

def matriz_zero(linhas, colunas):
  resultado = []
  for _ in range(linhas):
    linha = []
    for _ in range(colunas):
      linha.append(0)
    resultado.append(linha)

  return resultado

def matriz_zero_com_bordas(linhas, colunas, tamanho_borda, valor_borda):
    resultado = []

    for _ in range(linhas):
        linha = []
        for _ in range(tamanho_borda):
            linha.append(valor_borda)
        for _ in range(colunas):
            linha.append(0)
        for _ in range(tamanho_borda):
            linha.append(valor_borda)
        resultado.append(linha)

    linha = []
    for _ in range(colunas + tamanho_borda * 2):
        linha.append(valor_borda)

    for _ in range(tamanho_borda):
        resultado.append(linha)

    return np.array(resultado)


def matriz_to_string(matriz):
    borda_matriz = "| "
    resultado = ""

    maior_digito = 0
    for linha in matriz:
        for celula in linha:
            if len(str(celula)) > maior_digito:
                maior_digito = len(str(celula))

    for linha in matriz:
        resultado += borda_matriz
        for numero in linha:
            celula = str(numero)
            while len(celula) < maior_digito:
                celula = " " + celula
            resultado += celula + "  "
        resultado += borda_matriz + '\n'

    print(resultado)

def binarizar(vetor):
    for i in range(len(vetor)):
        vetor[i] = -1 if vetor[i] == -1 else (0 if vetor[i] == 0 else vetor[i])
    return vetor

def media(vetor):
    somatorio = 0
    for valor in vetor:
        somatorio += valor

    return somatorio
