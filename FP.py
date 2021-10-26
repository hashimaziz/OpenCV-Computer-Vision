#!/usr/bin/env python3

import numpy as np
import cv2
import time
import multiprocessing 
from multiprocessing import Pipe
import os
# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 30fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
font = cv2.FONT_HERSHEY_COMPLEX
face_cascade = cv2.CascadeClassifier("front_face.xml")
def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=820,
    display_height=616,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

def show_camera(conn, conn1, conn2):
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    print('parent: capture Image process id:',os.getppid())
    print('capture Image process id:',os.getpid())
    
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        # Window
        #cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

        while cv2.getWindowProperty("CSI Camera", 0) >= 0: 
            ret_val, img = cap.read()
            start_time = time.time()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgData = {'start_time': start_time}
            
            #print(FPSVal)
            
            conn1.send(gray)
            #print(conn1.poll())
            if conn1.poll():
                face = conn1.recv()
                print("recieved")
            else:
                face = ()
                print("skipped")
            #face = conn1.recv()
            #print(face)

            for (x, y, w, h) in face:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(img, "# of faces: " + str(len(face)), (10,600), font, 1, (0,255,0), 2, cv2.LINE_AA)
            
            keyCode = cv2.waitKey(30) & 0xFF

            # Stop the program on the ESC key
            if keyCode == 27:
                break
            
            imgData['end_time'] = time.time()
            conn.send(imgData)
            FPSVal = conn.recv()
            cv2.putText(img, "FPS: " + FPSVal, (10,40), font, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("CSI Camera", img)
            #print("cam")
    else:
        print("Unable to open camera")

    cap.release()
    cv2.destroyAllWindows()

def polling(conn, conn1):
    if conn.poll():
        face = conn.recv()
        print("Recieved")
    else:
        face = ()
        print("Skipped")
    conn1.send(face)

def faceRect(conn,):
    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        gray = conn.recv()
        face = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30,30)
            )
        #print(face)
        #print("face")
        conn.send(face)
        print("sent")
'''
def eyeRect(gray):
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30)
        )
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
'''

def FPS_Func(conn,):
    print('parent: FPS process id:',os.getppid())
    print('FPS process id:',os.getpid())
    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        ds = conn.recv()
        FPS = round((1.0/(ds['end_time'] - ds['start_time'])), 4)
        FPS = str(FPS)
        conn.send(FPS) 

face_cascade = cv2.CascadeClassifier('front_face.xml')

processes=[]
parent_conn, child_conn = Pipe()
face_conn, scFace = Pipe()

p = multiprocessing.Process(target=show_camera, args = (child_conn,face_conn)) 
processes.append(p)
p.start()

p = multiprocessing.Process(target=FPS_Func,args=(parent_conn,))
processes.append(p)
p.start()

p = multiprocessing.Process(target=faceRect, args=(scFace,))
processes.append(p)
p.start()

p = multiprocessing.Process(target=polling, args=())
processes.append(p)
p.start()

for p in processes: p.join()