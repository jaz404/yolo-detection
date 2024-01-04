import cv2
from ultralytics import YOLO
import math


import cvzone

#cap = cv2.VideoCapture(0) # webcam: 0 if no external webcam
#cap.set(3,1280)
#cap.set(4, 720)

cap = cv2.VideoCapture("../Videos/ppe-2.mp4")

model = YOLO('ppe.pt')

classNames = ['Hardhat','Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'machinery','vehicle']

myColor = (0,0,255)
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        # to get all the bounded boxes
        boxes =r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # 2 methods(formats) x1 y1 and x2 y2 or x1 y1 and width and height
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),3) # image, points
            # (0,0,0), 3 are the color and thickness

            w, h = x2-x1, y2-y1
            #cvzone.cornerRect(img, (x1, y1, w, h))


            # to get the confidence level:

            conf = (math.ceil(box.conf[0]*100))/100 # to get 2 decimal places

            # identifying classnames
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf>0.5:
                if currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == 'Mask':
                    myColor=(0,255,0)

                elif currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == 'NO-Mask':

                    myColor=(0,0,255)

                else:
                    myColor=(255,0,0)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',(max(0, x1), max(35, y1)), scale=0.5, thickness=1, colorB=myColor, colorT=(255,255,255), colorR=myColor)

                cv2.rectangle(img, (x1,y1), (x2,y2), myColor,3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)