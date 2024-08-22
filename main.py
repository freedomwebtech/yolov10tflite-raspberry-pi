import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from vidgear.gears import CamGear

model = YOLO('best_float32.tflite')  


stream = CamGear(source='https://youtu.be/zbmKmLPZ_Hw', stream_mode = True, logging=True).start() # YouTube Video URL as input

cap=cv2.VideoCapture('t1.mp4')
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count=0

   
        
while True:
    frame = stream.read()
    if frame is None:
        break
    
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model(frame,imgsz=240)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    car=[]
    motorcycle=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        cvzone.putTextRect(frame,f'{c}',(x5,y5),1,1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
         
  

    cv2.imshow("FRAME", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()

