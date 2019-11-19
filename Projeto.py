import numpy as np
import cv2


def equaliza_histograma(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))


def limiarizar(img):
    ret, limiar = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
    return limiar


def binarizar_YCrCb(image):
    img = equaliza_histograma(image)

    piso = np.array([45, 35, 175], dtype=np.uint8)
    teto = np.array([245, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(img, piso, teto)
    mask = cv2.GaussianBlur(mask, (15, 15), 100)

    return mask


def show(image):
    cv2.imshow("imagem", image)
    cv2.waitKey(0)


def tentativa2(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    img = binarizar_YCrCb(img)
    mask = limiarizar(img)
    kernel = np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]])
    mask = cv2.dilate(mask, kernel, iterations=7)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.Canny(mask, 100, 200)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    areas = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        width = boundRect[i][2]
        height = boundRect[i][3]
        areas[i] = width*height
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    height, width = image.shape[:-1]
    mask1 = np.zeros((height + 2, width + 2), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255, 0, 0)
        cv2.drawContours(drawing, contours_poly, i, color)
        if areas[i] > 80000:
            cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])),
                          (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
                          (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
    cv2.floodFill(drawing, mask1, (0,0), (255,255,255))

    return drawing


camera = cv2.VideoCapture(0)


def captura():
    a, img = camera.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # conversão para YCrCb
    out = tentativa2(img)

    return img, out


def main():
    while True:
        img, out = captura()
        cv2.imshow("olhaso", img)
        cv2.imshow("outro", out)
        if cv2.waitKey(1) == 27:
            break


main()
