import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from tracker import Tracker
from vidgear.gears import CamGear

model = YOLO('best_float32.tflite')  


stream = CamGear(source='https://youtu.be/zbmKmLPZ_Hw', stream_mode = True, logging=True).start() # YouTube Video URL as input

cap=cv2.VideoCapture('t1.mp4')
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker=Tracker()
tracker1=Tracker()

count=0
cy1=226
offset=10

car_counter=[]

motorcycle_counter=[]    
   
        
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
        if 'car' in c:
            car.append([x1,y1,x2,y2])

        if 'motorcycle' in c:
            motorcycle.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(car)
    bbox_idx1=tracker1.update(motorcycle)
   
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if cy1<(cy+offset) and cy1>(cy-offset):
           cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
           cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
           cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
           if car_counter.count(id)==0:
               car_counter.append(id)
    for bbox1 in bbox_idx1:
        x5,y5,x6,y6,id1=bbox1
        cx3=int(x5+x6)//2
        cy3=int(y5+y6)//2
        if cy1<(cy3+offset) and cy1>(cy3-offset):
           cv2.circle(frame,(cx3,cy3),4,(255,0,255),-1)
           cvzone.putTextRect(frame,f'{id1}',(x5,y5),1,1)
           cv2.rectangle(frame,(x5,y5),(x6,y6),(255,0,0),2)
           if motorcycle_counter.count(id1)==0:
              motorcycle_counter.append(id1)
           
          
    ccounter=len(car_counter)
    mcounter=len(motorcycle_counter)
    cv2.line(frame,(236,226),(1018,226),(0,0,255),1)
    cvzone.putTextRect(frame,f'Carcounter:-{ccounter}',(60,50),1,1)
    cvzone.putTextRect(frame,f'Mcounter:-{mcounter}',(60,150),1,1)


    cv2.imshow("FRAME", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()

