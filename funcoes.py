import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math

def skin_detection_minimun_facial_features(imagem):
    nova_imagem = np.copy(imagem)
    nova_imagem = np.float32(nova_imagem) / 255

    R = nova_imagem[:, :, 0]
    G = nova_imagem[:, :, 1]
    B = nova_imagem[:, :, 2]
    result = B
    for i in range(0, len(nova_imagem)):
        for j in range(0, len(nova_imagem[0])):
            r = R[i][j] / R[i][j] + G[i][j] + B[i][j]
            g = G[i][j] / R[i][j] + G[i][j] + B[i][j]

            # limite superior
            F1 = -1.367 * math.pow(r, 2) + 1.0743 * r + 0.2
            # limite inferior
            F2 = -0.776 * math.pow(r, 2) + 0.5601 * r + 0.18

            w = math.pow((r - 0.33), 2) + math.pow((g - 0.33), 2) > 0.001

            up = (R[i][j] - (G[i][j] / 2 + B[i][j] / 2))

            down = math.sqrt((R[i][j] - G[i][j]) ** 2 + (R[i][j] - B[i][j]) * (G[i][j] - B[i][j]))

            teta = math.acos(up / math.sqrt(down))

            if B[i][j] > G[i][j]:
                H = 360.0 - teta
            H = teta

            if np.all(np.logical_and((np.logical_and((np.logical_and((g < F1), (g > F2))), (w > 0.001))),
                                     (np.logical_or((H > 240), (H <= 20))))):
                skin = 1
            else:
                skin = 0

            result[i][j] = H

    return (result * 255).astype(np.uint8)


def skin_detection(imagem):
    # define os limites superior e inferior do pixel HSV
    # intensidades a serem consideradas 'pele'
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    converted = cv.cvtColor(imagem, cv.COLOR_BGR2HSV)
    skinMask = cv.inRange(converted, lower, upper)
    # aplicação de uma série de erosões e dilatações na máscara
    # usando um kernel elíptico
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    skinMask = cv.erode(skinMask, kernel, iterations=2)
    skinMask = cv.dilate(skinMask, kernel, iterations=2)
    # desfoque a máscara para ajudar a remover o ruído e, em seguida, aplique o
    # máscara para a moldura
    skinMask = cv.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv.bitwise_and(imagem, imagem, mask=skinMask)
    # mostra a pele da imagem junto com a máscara
    return imagem, skin
    
def RGB_color_space(imagem):
    r, g, b = cv.split(imagem)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = imagem.reshape((np.shape(imagem)[0] * np.shape(imagem)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Vermelho")
    axis.set_ylabel("Verde")
    axis.set_zlabel("Azul")
    plt.savefig('imagens/saida/image_in_rgb_color_space.png')
    return True


def HSV_color_space(imagem):
    nova_imagem = cv.cvtColor(imagem, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(nova_imagem)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = imagem.reshape((np.shape(imagem)[0] * np.shape(imagem)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Matiz")
    axis.set_ylabel("Saturação")
    axis.set_zlabel("Brilho")
    plt.savefig('imagens/saida/image_in_hsv_color_space.png')
    return True