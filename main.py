import cv2 as cv
import numpy as np
import funcoes
import minimun_facial_features

canais = ['R', 'G', 'B', 'S', 'I', 'H']

img = cv.imread('imagens/entrada/img_1.png', 1)
img = cv.resize(img, (500, 500))
# SKIN
skin = minimun_facial_features.skin_detection(img)
cv.imshow('SKIN DETECTION', skin)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/skin.png', skin)
quantization_skin = minimun_facial_features.skin_quantization(skin)
cv.imshow('QUANTIZATION SKIN', quantization_skin)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_skin.png', quantization_skin)
# HAIR
hair = minimun_facial_features.hair_detection(img)
cv.imshow('HAIR DETECTION', hair)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/hair.png', hair)
quantization_hair = minimun_facial_features.hair_quantization(hair)
cv.imshow('QUANTIZATION HAIR', quantization_hair)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_hair.png', quantization_hair)
# TODO: FUSION OF FEATURES
