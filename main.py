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
quantization_skin_square = minimun_facial_features.pixel_square(quantization_skin)
cv.imshow('QUANTIZATION SKIN SQUARE', quantization_skin_square)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_skin_square.png', quantization_skin_square)
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
quantization_hair_square = minimun_facial_features.pixel_square(quantization_hair)
cv.imshow('QUANTIZATION HAIR SQUARE', quantization_hair_square)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_hair_square.png', quantization_hair_square)
# TODO: FUSION OF FEATURES
# SKIN
quantization_skin_square_labels = minimun_facial_features.image_components(quantization_skin_square)
cv.imshow('QUANTIZATION SKIN SQUARE LABELS', quantization_skin_square_labels)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_skin_square_labels.png', quantization_skin_square_labels)
quantization_skin_square_labels_cog = minimun_facial_features.center_of_gravity(quantization_skin_square_labels)
cv.imshow('QUANTIZATION SKIN SQUARE LABELS CENTER OF GRAVITY', quantization_skin_square_labels_cog)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_skin_square_labels_cog.png', quantization_skin_square_labels_cog)
quantization_skin_square_labels_cog_outer_points = minimun_facial_features.extreme_outer_points(
    quantization_skin_square_labels_cog)
cv.imshow('QUANTIZATION SKIN SQUARE LABELS CENTER OF GRAVITY WITH EXTREME OUTER POINTS',
          quantization_skin_square_labels_cog_outer_points)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_hair_square_labels_cog_outer_points.png',
           quantization_skin_square_labels_cog_outer_points)
# HAIR
quantization_hair_square_labels = minimun_facial_features.image_components(quantization_hair_square)
cv.imshow('QUANTIZATION HAIR SQUARE LABELS', quantization_hair_square_labels)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_hair_square_labels.png', quantization_hair_square_labels)
quantization_hair_square_labels_cog = minimun_facial_features.center_of_gravity(quantization_hair_square_labels)
cv.imshow('QUANTIZATION HAIR SQUARE LABELS CENTER OF GRAVITY', quantization_hair_square_labels_cog)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_hair_square_labels_cog.png', quantization_hair_square_labels_cog)
quantization_hair_square_labels_cog_outer_points = minimun_facial_features.extreme_outer_points(
    quantization_hair_square_labels_cog)
cv.imshow('QUANTIZATION HAIR SQUARE LABELS CENTER OF GRAVITY WITH EXTREME OUTER POINTS',
          quantization_hair_square_labels_cog_outer_points)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('imagens/saida/quantization_hair_square_labels_cog.png', quantization_hair_square_labels_cog_outer_points)
