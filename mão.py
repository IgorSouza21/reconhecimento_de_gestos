import numpy as np
import cv2

cap = cv2.VideoCapture(0) #objeto de captura

while(True):
	ret, frame = cap.read()
	kernel = np.ones((1,1),np.uint8)
        #definir região de interesse
	roi=frame[50:350, 50:350]
	cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
	hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

	#define range of skin color in HSV
	lower_skin = np.array([0,20,70], dtype=np.uint8)
	upper_skin = np.array([20,255,255], dtype=np.uint8)
	#extrair cor da pele da img
	mask = cv2.inRange(hsv, lower_skin, upper_skin)


	#dilatação para preencher os espaços em preto na mão
	mask = cv2.dilate(mask,kernel,iterations = 5)

    	#borrar a img
	mask = cv2.GaussianBlur(mask,(5,5),100)

	#Mostra a img
	cv2.imshow('Captura',mask)
	cv2.imshow('Captura0', frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
#quando acabar o interesse, liberte a captura e encerre
cap.release()
cv2.destroyAllWindows()

