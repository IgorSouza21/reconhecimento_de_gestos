import cv2
import numpy as np

cap = cv2.VideoCapture(0)
kernel = np.ones((3, 3), np.uint8)


def equaliza_histograma(img):
    Y, Cr, Cb = cv2.split(img)
    y = cv2.equalizeHist(Y)
    cr = cv2.equalizeHist(Cr)
    cb = cv2.equalizeHist(Cb)
    return cv2.merge((y, cr, cb))


def mostrarImagem(nomeDaTela, img):
    cv2.imshow(nomeDaTela, img)


def limiarizar(img):
    imagem = img
    ret, limiar = cv2.threshold(imagem, 100, 255, cv2.THRESH_TOZERO)
    return limiar


def binarizar_YCrCb(entrada):
    img = entrada
    #apliico um filtro de mediana para remover a maior parte dos ruídos
    img = cv2.medianBlur(img, 13)
    #threshold
    img = limiarizar(img)

    # #Intervalo de cores em YCrCb para pele (foi calibrado no sangue, suor, lágrimas e tristeza)
    # piso = np.array([0, 60, 75], dtype=np.uint8)
    # teto = np.array([245, 145, 135], dtype=np.uint8)

    # Intervalo de cores em YCrCb para pele (foi calibrado no sangue, suor, lágrimas e tristeza)
    piso = np.array([0, 60, 100], dtype=np.uint8)
    teto = np.array([245, 175, 135], dtype=np.uint8)

    mask = cv2.inRange(img, piso, teto)
    #isso aqui remove a maior parte dos ruidos

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # mask = cv2.GaussianBlur(mask, (3, 3), 5)

    return 255-mask


def main2():
    while True:
        ret, imagem = cap.read()
        # imagem = cv2.imread("peles.jpg")
        # imagem = cv2.resize(imagem,(640,480))

        img = cv2.cvtColor(imagem, cv2.COLOR_RGB2YCrCb)
        # img = equaliza_histograma(img)

        mostrarImagem("Original",imagem)

        # mostrarImagem("Limiar", limiarizar(img))

        bin = binarizar_YCrCb(img)
        mostrarImagem("Bin", bin)

        # canny = cv2.Canny(imagem, 100,200)
        # mostrarImagem("meadian", canny)


        ret, labels = cv2.connectedComponents(bin)
        # Associa os rótulos a cores
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # HSV para BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        mostrarImagem("isto", labeled_img)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            print(img.shape)
            cap.release()
            break


main2()
cv2.destroyAllWindows()
