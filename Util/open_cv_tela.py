import cv2
import numpy as np

def plotar_features(layer):
    if layer == None:
        return
    # Assumindo que conv1_weights seja de tamanho (32, 4, 6, 6) - 32 filtros, 4 canais, 6x6
    num_filters = layer.shape[1]  # Número de filtros (32)
    filter_height = layer.shape[2]  # Altura do filtro (6)
    filter_width = layer.shape[3]  # Largura do filtro (6)

    ncols = 8  # Número de colunas para mostrar os filtros
    nrows = (num_filters + ncols - 1) // ncols  # Calcular o número de linhas para acomodar todos os filtros

    # Cria uma imagem de canvas para mostrar os filtros
    canvas_height = filter_height * nrows
    canvas_width = filter_width * ncols
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)  # Imagem preta

    # Preenche a imagem de canvas com os filtros
    for i in range(num_filters):  # Iterar pelos filtros
        row = i // ncols  # Linha onde o filtro será colocado
        col = i % ncols   # Coluna onde o filtro será colocado

        filter_img = layer[0][i].detach().cpu().numpy()  # Pega o filtro i
        filter_img = cv2.normalize(filter_img, None, 0, 255, cv2.NORM_MINMAX)  # Normaliza para a faixa 0-255
        filter_img = np.uint8(filter_img)  # Converte para inteiro de 8 bits

        # Coloca o filtro na posição correta na imagem de canvas
        canvas[row * filter_height: (row + 1) * filter_height,
               col * filter_width: (col + 1) * filter_width] = filter_img

    # Exibe a imagem concatenada com os filtros
    cv2.imshow(f"Features", canvas)
    cv2.waitKey(10)