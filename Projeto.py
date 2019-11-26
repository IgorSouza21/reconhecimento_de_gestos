import numpy as np
import cv2
import math
# from principal import *


def equaliza_histograma(img):
    Y, Cr, Cb = cv2.split(img)
    y = cv2.equalizeHist(Y)
    cr = cv2.equalizeHist(Cr)
    cb = cv2.equalizeHist(Cb)
    return cv2.merge((y, cr, cb))


def mostrarImagem(nomeDaTela, img):
    cv2.imshow(nomeDaTela, img)


def limiarizar(img):
    imagem = img.copy()
    ret, limiar = cv2.threshold(imagem, 100, 255, cv2.THRESH_TOZERO)
    return limiar


def binarizar_YCrCb(ent, piso, teto):
    img = ent.copy()
    kernel = np.ones((3, 3), np.uint8)

    #apliico um filtro de mediana para remover a maior parte dos ruídos
    img = cv2.medianBlur(img, 13)
    #threshold
    img = limiarizar(img)

    # #Intervalo de cores em YCrCb para pele (foi calibrado no sangue, suor, lágrimas e tristeza)
    # piso = np.array([0, 60, 75], dtype=np.uint8)
    # teto = np.array([245, 145, 135], dtype=np.uint8)

    # Intervalo de cores em YCrCb para pele (foi calibrado no sangue, suor, lágrimas e tristeza)
    # piso = np.array([0, 60, 100], dtype=np.uint8)
    # teto = np.array([245, 175, 135], dtype=np.uint8)

    mask = cv2.inRange(img, piso, teto)
    #isso aqui remove a maior parte dos ruidos

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=3)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # mask = cv2.GaussianBlur(mask, (3, 3), 5)

    return 255-mask


def show(image):
    cv2.imshow("imagem", image)
    cv2.waitKey(0)


def tentativa2(image, piso, teto):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    mask = binarizar_YCrCb(img, piso, teto)
    # mask = limiarizar(img)
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=2)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.Canny(mask, 100, 200)

    # ret, labels = cv2.connectedComponents(mask)
    # # Associa os rótulos a cores
    # label_hue = np.uint8(179 * labels / np.max(labels))
    # blank_ch = 255 * np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    #
    # # HSV para BGR
    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    cnt_poly = cv2.approxPolyDP(cnt, 3, True)
    bound_rect = cv2.boundingRect(cnt_poly)
    color = (255, 0, 0)
    # cv2.drawContours(mask, cnt_poly, 0, color, cv2.FILLED)
    cv2.rectangle(image, (int(bound_rect[0]), int(bound_rect[1])),
                  (int(bound_rect[0] + bound_rect[2]), int(bound_rect[1] + bound_rect[3])), color, 2)

    # cv2.floodFill(drawing, mask1, (int(boundRect[i][0]), int(boundRect[i][1])), (0, 0, 0))
    # cv2.floodFill(drawing, mask1, (0, 0), (0, 0, 0))

    # return labeled_img
    return mask


def tentativa3(frame):
    kernel = np.ones((3, 3), np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # extract skin colur imagw
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask, kernel, iterations=4)

    # blur the image
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    areas = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        width = boundRect[i][2]
        height = boundRect[i][3]
        areas[i] = width * height
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # height, width = image.shape[:-1]
    # mask1 = np.zeros((height + 2, width + 2), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255, 0, 0)
        # for con in contours_poly:
        #     cv2.drawContours(drawing, [con], -1, (0, 0, 255), cv2.FILLED)
        cv2.drawContours(drawing, contours_poly, i, color, cv2.FILLED)
        if areas[i] > 75000:
            cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])),
                          (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
                          (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

    return drawing
# camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# # Check if the webcam is opened correctly
# if not camera.isOpened():
#     raise IOError("Cannot open webcam")
#
#
# def captura():
#     a, img = camera.read()
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # conversão para YCrCb
#     out = tentativa2(img)
#
#     return img, out
#
#
# def main():
#     while True:
#         img, out = captura()
#         cv2.imshow("olhaso", img)
#         cv2.imshow("outro", out)
#         if cv2.waitKey(1) == 27:
#             break


# main()
# cv2.destroyAllWindows()