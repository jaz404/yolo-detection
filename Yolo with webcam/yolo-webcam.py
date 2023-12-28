import cv2
from ultralytics import YOLO
import math


import cvzone

# cap = cv2.VideoCapture(0) # webcam: 0 if no external webcam
# cap.set(3,1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("../Videos/bikes.mp4")

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
            cvzone.cornerRect(img, (x1, y1, w, h))

            # to get the confidence level:

            conf = (math.ceil(box.conf[0]*100))/100 # to get 2 decimal places

            # identifying classnames
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',(max(0, x1), max(35, y1)), scale=1, thickness=1)


    cv2.imshow("Image", img)
    cv2.waitKey(1)