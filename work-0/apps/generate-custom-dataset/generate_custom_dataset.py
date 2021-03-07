# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2, time
import numpy as np
import matplotlib.pyplot as plt
import time

# OpenCV Constants 
#=======================================================================================
CAM = cv2.VideoCapture(0)


ROI_H = [90,390]
ROI_W = [170,470]

SKIN_LO = np.array([0,48,80], dtype=np.uint8)
SKIN_HI = np.array([20,255,255], dtype=np.uint8)

RED     = (0, 0, 255)
GREEN   = (0, 255, 0)
BLUE    = (255, 0, 0)
ORANGE  = (0, 69, 255)
MAGENTA = (240, 0, 159)

FONT = cv2.FONT_HERSHEY_SIMPLEX

nothing = lambda nothing: None

cv2.namedWindow("Generate Custom Dataset")
cv2.createTrackbar('Save','Generate Custom Dataset',0,1,nothing)
cv2.createTrackbar('Label','Generate Custom Dataset',0,9,nothing)

# Useful Functions
#=======================================================================================
counter = np.ones(10) * 200

def img_write(img, label):
    if counter[label]:
        i = int(counter[label])
        filename = f"./custom_dataset/fig{label}/img-{label}-{i}.png"
        counter[label] -=1
        cv2.imwrite(filename, img)
        print(f"Figure-{label} images left: {i}")
    else:
        print(f"Figure-{label} COMPLETE!")

def detect_skin(FRAME):
    roi = FRAME[ROI_H[0]:ROI_H[1],ROI_W[0]:ROI_W[1]]
    FRAME = cv2.blur(FRAME, (20,20))
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(roi_hsv, SKIN_LO, SKIN_HI)
    skinMask = cv2.dilate(skinMask, None, iterations=1)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(roi, roi, mask = skinMask)
    FRAME[ROI_H[0]:ROI_H[1],ROI_W[0]:ROI_W[1]] = skin
    return FRAME, skin


# Main Loop
#=======================================================================================
while True:
    if cv2.waitKey(1) == 27:
        camera.release()
        cv2.destroyAllWindows()
        break

    SAVE  = cv2.getTrackbarPos("Save","Generate Custom Dataset")
    LABEL = cv2.getTrackbarPos("Label","Generate Custom Dataset")

    _, FRAME = CAM.read()
        
    FRAME, IMG = detect_skin(FRAME)

    if SAVE:
        time.sleep(0.2)
        img_write(IMG, LABEL)
      
    cv2.imshow("Generate Custom Dataset", FRAME)   

