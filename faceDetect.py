import cv2
import numpy as np 

#Loading the classifier
faceDetect = cv2.CascadeClassifier('harcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

while(true):
	ret, img = cam.read()

	#Converting to Grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Detect Faces
	faces = faceDetect.DetectMultiScale(gray, 1.3, 5)

	#Drawing a rectangle
	for(x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

	cv2.imshow("Faces", img)

k = cv2.waitKey(100) & 0xff 
if k==27:
	break

cam.release()
cam.destroyAllWindows()
