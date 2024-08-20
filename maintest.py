import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np

model = YOLO('best_float32.tflite')

cap = cv2.VideoCapture('p1.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
while True:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))

    results = model(frame, imgsz=240)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    boxes = []
    confidences = []
    class_ids = []
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        confidence = float(row[4])
        class_id = int(row[5])
        
        # Only process boxes related to 'person' class
        if class_list[class_id] == 'person':
            boxes.append([x1, y1, x2, y2])
            confidences.append(confidence)
            class_ids.append(class_id)
    
    # Ensure boxes and confidences are not empty
    if len(boxes) == 0:
        continue
    
    # Apply Non-Maximum Suppression
    boxes_np = np.array(boxes)
    confidences_np = np.array(confidences)
    fbox = cv2.dnn.NMSBoxes(boxes_np.tolist(), confidences_np.tolist(), score_threshold=0.1, nms_threshold=0.4)
    
    if len(fbox) > 0:
        fbox = fbox.flatten()
        for i in fbox:
            box = boxes[i]
            x1, y1, x2, y2 = box
            class_id = class_ids[i]
            c = class_list[class_id]
            
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    cv2.imshow("FRAME", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
