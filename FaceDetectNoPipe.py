import numpy as np 
import cv2
import time

font = cv2.FONT_HERSHEY_COMPLEX
face_cascade = cv2.CascadeClassifier("front_face.xml")
eye_cascade = cv2.CascadeClassifier("eye.xml")
cv2.namedWindow("Cam", cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)
keypress = None

def classification(file, gray):
    list = file.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30)
    )
    return list

def drawRect(list, color):
    for(x,y,w,h) in list:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)

while(keypress != 27):
    start_time = time.time()
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face = classification(face_cascade, gray)
    eyes = classification(eye_cascade, gray)
    drawRect(face, (0,255,0))
    drawRect(eyes, (0,0,255))

    cv2.putText(img, "# of faces: " + str(len(face)), (10,600), font, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(img, "FPS: " + str(round((1.0/(time.time() - start_time)), 4)), (0,40), font, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Cam", img)
    keypress = cv2.waitKey(30) & 0xFF

cap.release()
