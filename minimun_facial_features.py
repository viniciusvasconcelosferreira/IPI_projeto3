import math
import numpy as np
import cv2 as cv


def color_space_trans(imagem, par):
    epselon = 0.00001
    nova_imagem = np.copy(imagem)
    nova_imagem = np.float32(nova_imagem) / 255

    R = nova_imagem[:, :, 0]
    G = nova_imagem[:, :, 1]
    B = nova_imagem[:, :, 2]

    if par == 'R':
        result = R

    if par == 'G':
        result = G

    if par == 'B':
        result = B

    if par == 'I':
        inte = R + G + B
        inte = inte / 3
        result = inte

    if par == 'S':
        ss = R + G + B + epselon
        s = 1 - (3 * np.minimum(np.minimum(R, G), B) / ss)
        result = s

    if par == 'H':
        result = B
        for i in range(0, len(nova_imagem)):
            for j in range(0, len(nova_imagem[0])):
                up = (R[i][j] - (G[i][j] / 2 + B[i][j] / 2))

                down = math.sqrt((R[i][j] - G[i][j]) ** 2 + (R[i][j] - B[i][j]) * (G[i][j] - B[i][j]))

                teta = math.acos(up / math.sqrt(down))

                if B[i][j] > G[i][j]:
                    H = 360.0 - teta
                H = teta

                result[i][j] = H
    return (result * 255).astype(np.uint8)


def skin_detection(imagem):
    global skin, F1, F2
    nova_imagem = np.copy(imagem)
    nova_imagem = np.float32(nova_imagem) / 255

    R = nova_imagem[:, :, 0]
    G = nova_imagem[:, :, 1]
    B = nova_imagem[:, :, 2]
    result = B
    for i in range(0, len(nova_imagem)):
        for j in range(0, len(nova_imagem[0])):
            # normalização do modelo RGB
            r = R[i][j] / R[i][j] + G[i][j] + B[i][j]
            g = G[i][j] / R[i][j] + G[i][j] + B[i][j]

            # limite superior
            F1 = -1.367 * r ** 2 + 1.0743 * r + 0.2
            # limite inferior
            F2 = -0.776 * r ** 2 + 0.5601 * r + 0.18
            # exclusão da cor branca
            w = ((r - 0.33) ** 2 + (g - 0.33) ** 2) > 0.001

            up = (2 * R[i][j] - G[i][j] - B[i][j]) / 2

            down = math.sqrt((R[i][j] - G[i][j]) ** 2 + (R[i][j] - B[i][j]) * (G[i][j] - B[i][j]))

            teta = math.acos(up / math.sqrt(down))

            if B[i][j] > G[i][j]:
                H = 360 - teta
            else:
                H = teta

            if np.all(np.logical_and((np.logical_and((np.logical_and((g < F1), (g > F2))), (w > 0.001))),
                                     (np.logical_or((H > 240), (H <= 20))))):
                skin = 1
            else:
                skin = 0

            result[i][j] = H
    return (result * 255).astype(np.uint8)


def hair_detection(imagem):
    global hair, H

    nova_imagem = np.copy(imagem)
    nova_imagem = np.float32(nova_imagem) / 255

    R = nova_imagem[:, :, 0]
    G = nova_imagem[:, :, 1]
    B = nova_imagem[:, :, 2]

    I = (R + G + B) / 3

    if np.all(np.logical_or((np.logical_and((I < 80), (np.logical_or((B - G < 15), (B - R < 15))))),
                            (20 < H <= 40))):
        hair = 1
    else:
        hair = 0

    return I


def skin_quantization():
    return


def hair_quantization():
    return
