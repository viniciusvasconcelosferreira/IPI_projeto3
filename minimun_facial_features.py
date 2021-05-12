import math
import numpy as np
import cv2 as cv
from sklearn.cluster import MiniBatchKMeans


def binary_image_skin(imagem):
    if len(imagem.shape) > 2:
        imagem = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    suave = cv.GaussianBlur(imagem, (7, 7), 0)  # aplica blur
    _, bin = cv.threshold(suave, 160, 255, cv.THRESH_BINARY)
    cv.imwrite('imagens/saida/binary_image_skin_THRESH_BINARY.png', bin)


def binary_image_hair(imagem):
    if len(imagem.shape) > 2:
        imagem = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    suave = cv.GaussianBlur(imagem, (7, 7), 0)  # aplica blur
    _, binI = cv.threshold(suave, 160, 255, cv.THRESH_BINARY_INV)
    cv.imwrite('imagens/saida/binary_image_hair_THRESH_BINARY_INV.png', binI)


def binary_image(imagem):
    if len(imagem.shape) > 2:
        imagem = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    suave = cv.GaussianBlur(imagem, (7, 7), 0)  # aplica blur
    _, bin = cv.threshold(suave, 160, 255, cv.THRESH_BINARY)
    _, binI = cv.threshold(suave, 160, 255, cv.THRESH_BINARY_INV)
    cv.imwrite('imagens/saida/binary_image_suavizada.png', suave)
    cv.imwrite('imagens/saida/binary_image_THRESH_BINARY.png', bin)
    cv.imwrite('imagens/saida/binary_image_THRESH_BINARY_INV.png', binI)
    cv.imwrite('imagens/saida/binary_image_with_mask.png', cv.bitwise_and(imagem, imagem, mask=binI))


def fill_outline_skin(imagem):
    thresh = cv.threshold(imagem, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv.fillPoly(imagem, cnts, [255, 255, 255])
    cv.imwrite('imagens/saida/binary_image_THRESH_BINARY_INV_fill_outline_skin.png', imagem)


def fill_outline_hair(imagem):
    thresh = cv.threshold(imagem, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv.fillPoly(imagem, cnts, [255, 255, 255])
    cv.imwrite('imagens/saida/binary_image_THRESH_BINARY_fill_outline_hair.png', imagem)


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
            # up = 0.5 * ((R[i][j] - G[i][j]) + (R[i][j] - B[i][j]))

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


def skin_quantization_binary(imagem):
    proporcao = 128  # Definir relação de quantização
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            for k in range(imagem.shape[2]):
                imagem[i][j][k] = int(imagem[i][j][k] / proporcao) * proporcao
    cv.imwrite('imagens/saida/binary_image_THRESH_BINARY_skin_quantization.png', imagem)


def skin_quantization(imagem):
    proporcao = 128  # Definir relação de quantização
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            imagem[i][j] = int(imagem[i][j] / proporcao) * proporcao
    return imagem


def skin_quantization_k_means_clustering(imagem):
    (h, w) = imagem.shape[:2]
    image = cv.cvtColor(imagem, cv.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = MiniBatchKMeans(8)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    quant = cv.cvtColor(quant, cv.COLOR_LAB2BGR)
    image = cv.cvtColor(image, cv.COLOR_LAB2BGR)
    cv.imwrite('imagens/saida/binary_image_THRESH_BINARY_skin_quantization_k_means.png', quant)


def hair_detection(imagem):
    global hair, H
    nova_imagem = np.copy(imagem)
    nova_imagem = np.float32(nova_imagem) / 255

    R = nova_imagem[:, :, 0]
    G = nova_imagem[:, :, 1]
    B = nova_imagem[:, :, 2]

    result = B
    for i in range(0, len(nova_imagem)):
        for j in range(0, len(nova_imagem[0])):
            I = (R[i][j] + G[i][j] + B[i][j]) / 3
            try:
                pixels_escuros = (I < 80)
                pixels_azul_profundo = np.logical_or((B - G < 15), (B - R < 15))
                pixels_marrom = (20 < H <= 40)
                if np.all(np.logical_or((np.all(np.logical_and(pixels_escuros, pixels_azul_profundo)) == True),
                                        (pixels_marrom))) == True:
                    hair = 1
                else:
                    hair = 0
            except:
                pass

            result[i][j] = I

    return (result * 255).astype(np.uint8)


def hair_quantization(imagem):
    proporcao = 128  # Definir relação de quantização
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            imagem[i][j] = int(imagem[i][j] / proporcao) * proporcao
    return imagem
