import cv2, time, multiprocessing, os
from multiprocessing import Pipe
from sys import platform

font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
face_cascade = cv2.CascadeClassifier("front_face.xml")
eye_cascade = cv2.CascadeClassifier("eye.xml")
cv2.namedWindow("Cam", cv2.WINDOW_AUTOSIZE)


def show_camera(conn,):
    keypress = None
    cap = cv2.VideoCapture(0)
    face = []
    eyes = []
    if(cap.isOpened()):
        while(keypress != 27):
            start_time = time.time()
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            conn.send(gray)

            if(conn.poll()): #if there is data to be recieved
                imgData = conn.recv()
                face = imgData['faces']
                eyes = imgData['eyes']
                for(x,y,w,h) in face:
                    cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 2)
                for(x,y,w,h) in eyes:
                    cv2.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 2)

            cv2.putText(img, "# of faces: " + str(len(face)), (10,600), font, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(img, "FPS: " + str(round((1.0/(time.time() - start_time)), 4)), (0,40), font, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("Cam", img)
            keypress = cv2.waitKey(30) & 0xFF
        
        os.system("killall -9 Python")
        cap.release()

def classification(conn,):
    keypress = None
    while(keypress != 27):
        if(conn.poll()): #if there is data to be received
            gray = conn.recv()
            face = face_cascade.detectMultiScale(
                gray,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (30,30)
            )
            eyes = eye_cascade.detectMultiScale(
                gray,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (30,30)
            )
            imgData = {'faces': face, 'eyes': eyes}
            conn.send(imgData)
        keypress = cv2.waitKey(30) & 0xFF

if(__name__ == "__main__"):
    if(platform == "darwin"):
        multiprocessing.set_start_method('spawn') #only on mac because Unix doesn't allow the 'fork' start method
    
    processes = []
    scConn, classConn = Pipe()
    
    p = multiprocessing.Process(target = show_camera, args = (scConn,))
    processes.append(p)
    p.start()

    p = multiprocessing.Process(target = classification, args = (classConn,))
    processes.append(p)
    p.start()

    for p in processes: p.join()