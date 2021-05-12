import cv2 as cv
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt


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
    # mostra a pele da imagem
    return skin


def face_detect(imagem):
    # carreguar a fotografia
    pixels = cv.imread('imagens/entrada/' + str(imagem) + '.png')
    # carreguar o modelo pré-treinado
    classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # realizar detecção de rosto
    bboxes = classifier.detectMultiScale(pixels)
    # imprimir caixa delimitadora para cada rosto detectado
    for box in bboxes:
        # extrair
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # desenhar um retângulo sobre os pixels
        cv.rectangle(pixels, (x, y), (x2, y2), (0, 255, 0), 1)
        # BGR
    # salva a imagem
    cv.imwrite('imagens/saida/' + str(imagem) + '_face_detect.png', pixels)


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
