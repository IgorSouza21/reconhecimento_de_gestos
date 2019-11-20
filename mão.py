import cv2
import numpy as np

def detect(img, cascade):
	rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5,minSize=(30,30),
									 flags=cv2.CASCADE_SCALE_IMAGE)
	if len(rects) == 0:
		return []
	rects[]