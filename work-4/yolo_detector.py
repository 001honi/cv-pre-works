from imutils.video import FPS
import imutils
import numpy as np 
import cv2
import os


inp = "videos/sample.mp4"
out = "videos/sample_YOLO.avi"

vs = cv2.VideoCapture(inp)

# YOLO
#===========================================================
conf_thresh = 0.5
nms_thresh = 0.3

configPath = "model/yolov3.cfg"
weightsPath = "model/yolov3.weights"
labelsPath = "model/coco.txt"

np.random.seed(42)
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS),3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
lyr_names = net.getLayerNames()
lyr_names = [lyr_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#===========================================================

# Total number of frames in video stream
try: 
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT 
    total = int(vs.get(prop)) 
except:
    total = -1

frame_id = 0

writer = False

while True:
    ret, frame = vs.read()

    if not (ret or frame):
        break

    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    if writer is False:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out, fourcc, 30, (W,H), True)

    fps = FPS().start()

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(lyr_names)

    boxes = [] ; confidences = [] ; classIDs = []

    for output in layerOutputs:
        for detection in output:
            # select the maximum-likely prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
    
            if confidence < conf_thresh:
                # if true other detections also lower confidence
                continue
            
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            # center coordinates to x1,y1
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            boxes.append([x, y, int(width), int(height)]) 
            confidences.append(float(confidence)) 
            classIDs.append(classID)

    # non-maxima-suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1]) 
            (w, h) = (boxes[i][2], boxes[i][3])     

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1) 
            text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i]) 
            cv2.putText(frame, text, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    fps.update()
    fps.stop()
    frame_id += 1
    cv2.putText(frame, f"FPS: {fps.fps():.2f}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(frame, f"{frame_id}/{total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    writer.write(frame)
    # cv2.imshow("YOLO", frame)
    if frame_id % 30 == 0:
        print( f"{frame_id}/{total}")

vs.release()
cv2.destroyAllWindows()