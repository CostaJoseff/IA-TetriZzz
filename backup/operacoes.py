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

    return resultado


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
