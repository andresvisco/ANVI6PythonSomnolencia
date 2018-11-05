# USAGE
# python facial_landmarks.py


from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import requests 
import asyncio
import _thread

from imutils.video import VideoStream
import matplotlib.pyplot as plt
import json

async def enviar(stringValor):
    response=requests.post(url, data=data_json, headers=headers)


tiempoOjoAbierto=0
tiempoOjoCerrado=0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# detect faces in the grayscale image
#vs=VideoStream('rtsp://avisco:Sof7t3k!2018@192.168.180.185/0').start()
vs=VideoStream(0).start()
xaLimiteL=36
xaLimiteU=40
font = cv2.FONT_HERSHEY_SIMPLEX
headers = {'Content-type': 'application/json'}
url="[your powerbi]"


def enviarDatos(datosJson):
    response=requests.post(url, data_json, headers=headers)

while True:

    frame=vs.read()
    frame = imutils.resize(frame, width=500)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    rects = detector(gray, 0)
    xaP=[]
    xa=0
    suma=0
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #busco cada punto en el shape
        for(x,y) in shape:   
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            #aislo los 2 puntos correspondientes al ojo izquierdo (parpado arriba y abajo)
            if (xa>=xaLimiteL and xa<=xaLimiteU):
                if(xa%2!=0):                
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                    suma=x/y
                    
                    xaP+=[y]
                    
                    
                    if len(xaP)==2:
                        diferencia = abs(xaP[0] - xaP[1])
                        
                        if diferencia>4:
                            tiempoOjoAbierto+=1
                            cv2.putText(frame,"Ojos abiertos",(0,20),font, 0.6,(0,255,0),1, cv2.LINE_AA)
                        else:
                            tiempoOjoCerrado+=1
                            cv2.putText(frame,"Ojos cerrados",(0,20),font, 0.8,(0,0,255),1, cv2.LINE_AA)                            

                        cv2.putText(frame,str("Tiempo abierto: " + str(tiempoOjoAbierto)),(320,350),font, 0.4,(255,255,255),1, cv2.LINE_AA)                            
                        cv2.putText(frame,str("Tiempo cerrado: " + str(tiempoOjoCerrado)),(320,370),font, 0.4,(255,255,255),1, cv2.LINE_AA)                            
                        data={"abierto":tiempoOjoAbierto,"cerrado":tiempoOjoCerrado}
                        data_json=json.dumps(data)
                        print(_thread._count())
                        if(_thread._count()>4):
                            continue
                        else:
                            _thread.start_new_thread(enviarDatos, (data_json,))

                        
            else:
                xaP=[]
            xa+=1
        cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        
        cv2.destroyAllWindows()
        vs.stop()
        break




