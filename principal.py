import cv2
import numpy as np

cap = cv2.VideoCapture(1)
kernel = np.ones((3, 3), np.uint8)

def equaliza_histograma(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))

def mostrarImagem(nomeDaTela, img):
    cv2.imshow(nomeDaTela, img)

def limiarizar(img):
    ret, limiar = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
    return limiar
def binarizar_YCrCb(entrada):
    img = equaliza_histograma(entrada)
    # img = limiarizar(img)
    # Y, Cr, Cb = cv2.split(img)
    # mostrarImagem("Y", Y)
    # mostrarImagem("Cr", Cr)
    # mostrarImagem("Cb", Cb)
    mostrarImagem("YCrCb", img)
    piso = np.array([0, 20, 170], dtype=np.uint8)
    teto = np.array([245, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(img, piso, teto)
    # for i in range(10):
        # mask = cv2.erode(cv2.dilate(mask, kernel, iterations=1), kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (15, 15), 100)

    return mask

def main():
    while (True):
        ret, imagem = cap.read()
        # imagem = cv2.imread("peles.jpg")
        # imagem = cv2.resize(imagem,(480,640))
        img = cv2.cvtColor(imagem, cv2.COLOR_RGB2YCrCb)

        img = equaliza_histograma(img)

        mostrarImagem("Original",imagem)

        mostrarImagem("Limiar", limiarizar(img))

        piso = np.array([0, 20, 170], dtype=np.uint8)
        teto = np.array([245, 255, 255], dtype=np.uint8)

        mostrarImagem("Bin", binarizar_YCrCb(img))



        if(cv2.waitKey(1) & 0xFF == ord('q')):
            print(img.shape)
            cap.release()
            break

main()
cv2.destroyAllWindows()