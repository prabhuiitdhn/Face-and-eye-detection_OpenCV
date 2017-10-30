#improrting the packages
import numpy as np
import cv2
import argparse

#cascade xml for face and eyes
face_cascade=cv2.CascadeClassifier('/home/cyrrup/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('/home/cyrrup/opencv/data/haarcascades/haarcascade_eye.xml')

#argument pass for images
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
	help="Please give the image path")

args=vars(ap.parse_args())
face_image=cv2.imread(args["image"])
gray=cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray, 1.3,5)

for(x, y, w, h) in faces:
	cv2.rectangle(face_image,(x,y),(x+w, y+h), (255, 0, 0), 2)
	roi_gray=gray[y:y+h, x:x+w]
	roi_color=face_image[y:y+h, x:x+w]
	eyes=eye_cascade.detectMultiScale(roi_gray)
	for(ex, ey, ew, eh) in eyes:
		cv2.rectangle(roi_color,(ex, ey), (ex+ew, ey+eh),(0,255,0),2)

cv2.imshow('Face_image',face_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


