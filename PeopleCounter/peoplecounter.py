import cv2
import numpy as np
from ultralytics import YOLO
import math

from sort import* # required for creating ids and counting unique bodies
import cvzone




cap = cv2.VideoCapture("../Videos/people.mp4")

model = YOLO('../yolo-weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# make the mask layer
mask = cv2.imread("../Videos/people-mask.png")

# implementing the tracker

tracker = Sort(max_age=20, min_hits =2, iou_threshold=0.3) #max_age is the no. of frames before it considers it as a new object
# iou is the intersection of union = area of overlap / area of union

# 2 lines one for people going up and one for people going down

lineUp = [103, 161, 296, 161]
lineDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask) # this is the only required region which we get on doing and on img and mask

    imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(730,260))


    results = model(imgRegion, stream=True) # replace with imgRegion to get results from masked video


    detections = np.empty((0,5))# numpy array of dectections for the tracker
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
            #cvzone.cornerRect(img, (x1, y1, w, h),l =8, rt = 5)

            # to get the confidence level:

            conf = (math.ceil(box.conf[0]*100))/100 # to get 2 decimal places

            # identifying classnames
            cls = int(box.cls[0])

            currentCls = classNames[cls]

            # the ideal location of detection is not too far or too close

            if currentCls == 'person' and conf > 0.3: # only runs if it is a person and if confidence is > than

                #cvzone.putTextRect(img, f'{classNames[cls]} {conf}',(max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))


    resultsTracker = tracker.update(detections)
    # cv2.line(img,(line[0],line[1]),(line[2],line[3]),(0,0,255),5)

    cv2.line(img,(lineUp[0],lineUp[1]),(lineUp[2],lineUp[3]),(0,0,255),5)
    cv2.line(img, (lineDown[0], lineDown[1]), (lineDown[2], lineDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=1, offset=10)


        # finding the center of the boxes to check if it crossed line

        cx,cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx,cy),2,(0,255,0),cv2.FILLED)

        # if it crosses:
        if lineUp[0] < cx < lineUp[2] and lineUp[1]-20 < cy < lineUp[1]+20:
            if Id not in totalCountUp:
                totalCountUp.append(Id)
                cv2.line(img, (lineUp[0], lineUp[1]),(lineUp[2],lineUp[3]), (0,0,255), 5)

        if lineDown[0] < cx < lineDown[2] and lineDown[1]-20 < cy < lineDown[1]+20:
            if Id not in totalCountDown:
                totalCountDown.append(Id)
                cv2.line(img, (lineDown[0], lineDown[1]),(lineDown[2],lineDown[3]), (0,0,255), 5)
    #     #cvzone.putTextRect(img, f'cars: {len(totalCount)}', (50, 50), scale=2, thickness=1, offset=10)
    #
    cv2.putText(img, str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),8)
    cv2.putText(img, str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),8)

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)

    cv2.waitKey(1)