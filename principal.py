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
    # imagem = equaliza_histograma(img)
    ret, limiar = cv2.threshold(imagem, 100, 255, cv2.THRESH_TOZERO)

    return limiar

def binarizar_YCrCb(entrada):
    img = entrada
    img = limiarizar(img)


    piso = np.array([0, 35, 100], dtype=np.uint8)
    teto = np.array([245, 175, 135], dtype=np.uint8)

    mask = cv2.inRange(img, piso, teto)
    # mask = cv2.dilate(mask,kernel, iterations=5)
    # mask = cv2.erode(mask,kernel,iterations=5)
    cv2.imshow("mask",mask)

    # for i in range(10):
        # mask = cv2.erode(cv2.dilate(mask, kernel, iterations=1), kernel, iterations=2)
    # mask = cv2.GaussianBlur(mask, (15, 15), 15)

    return mask

def main():
    while (True):
        ret, imagem = cap.read()
        # imagem = cv2.imread("peles.jpg")
        # imagem = cv2.resize(imagem,(480,640))

        img = cv2.cvtColor(imagem, cv2.COLOR_RGB2YCrCb)
        # img = equaliza_histograma(img)

        mostrarImagem("Original",imagem)

        mostrarImagem("Limiar", limiarizar(img))


        mostrarImagem("Bin", binarizar_YCrCb(img))



        if(cv2.waitKey(1) & 0xFF == ord('q')):
            print(img.shape)
            cap.release()
            break

main()
cv2.destroyAllWindows()