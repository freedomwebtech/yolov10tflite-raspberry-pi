import cv2
import time
cpt = 0
maxFrames = 40 # if you want 5 frames only.

cap=cv2.VideoCapture('p1.mp4')
count=0
while cpt < maxFrames:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
    frame=cv2.resize(frame,(1020,600))
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite("/home/pi/autodistil_rpi4_yolov8-main/images/img_%d.jpg" %cpt, frame)
    cpt += 1
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()