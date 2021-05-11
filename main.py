import cv2 as cv
import numpy as np

import minimun_facial_features

canais = ['R', 'G', 'B', 'S', 'I', 'H']
imagem = ['img_1']
for nome in imagem:
    # for canal in canais[:3]:
    img = cv.imread('imagens/entrada/' + str(nome) + '.png', 1)
    img = cv.resize(img, (500, 500))
    nova_imagem_skin = minimun_facial_features.skin_detection(img)
    cv.imwrite('imagens/saida/' + str(nome) + '_skin_detection_original.png', nova_imagem_skin)
    nova_imagem_hair = minimun_facial_features.hair_detection(img)
    cv.imwrite('imagens/saida/' + str(nome) + '_hair_detection_original.png', nova_imagem_hair)
