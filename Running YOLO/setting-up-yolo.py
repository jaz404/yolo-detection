from ultralytics import YOLO

# in order to see the result, we need opencv
import cv2

# creating the model
model = YOLO('../yolo-weights/yolov8l.pt') # yolov8n is the weight
# n = nano, l = large, m = medium
# medium, large slower
results = model("Images/3.png",show=True) # source of the image
# show = True shows the name of object and the confidence level
cv2.waitKey(0) # press anykey to quit
