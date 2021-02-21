import multiprocessing as mp
import numpy as np
import cv2
import os
import imutils
from imutils.video import FPS

tracker_type = "KCF"
max_tracker = 8
N = 60 # YOLO detection per number of frame
inp = "videos/sample.mp4"
out = f"videos/sample_YOLO_{N}_{tracker_type}.avi"

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
FONT = cv2.FONT_HERSHEY_SIMPLEX

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
lyr_names = net.getLayerNames()
lyr_names = [lyr_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# TRACKER
#===========================================================

def tracker_start(frame, box, id, tracker_id, inpQ, outQ):
    global tracker_type
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = cv2.TrackerKCF_create()
    tracker.init(frame, box)

    while True:
        frame = inpQ.get()
        if frame.any():
            (success, box) = tracker.update(frame)
            outQ.put((success, box, id, tracker_id))

# MAIN FUNCTION
#===========================================================

def main(vs, inpQueues, outQueues, N, total, writer=False, frame_id=0, fps_pre=0.0):
    while True:
        ret, frame = vs.read()
        if not (ret or frame):
            break

        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        if writer is False:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out, fourcc, 30, (W,H), True)

        if frame_id % N == 0:  
            # preprocessing
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            # get network output
            net.setInput(blob)
            layerOutputs = net.forward(lyr_names)
            # decision making
            boxes = [] ; confidences = [] ; classIDs = []
            for output in layerOutputs:
                for detection in output:
                    # select the maximum-likely prediction
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence < conf_thresh:
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

            if len(idxs):
                # reset trackers
                if inpQueues:
                    inpQueues = []; outQueues = []  

                tracker_id = 0
                for n_tracker, i in enumerate(idxs.flatten()):
                    if n_tracker == max_tracker:
                        break
                    (x, y) = (boxes[i][0], boxes[i][1]) 
                    (w, h) = (boxes[i][2], boxes[i][3])     
                    box = (x,y,w,h)
                    # start trackers
                    inpQ = mp.Queue()
                    outQ = mp.Queue()
                    inpQueues.append(inpQ)
                    outQueues.append(outQ)
                    # new daemon process for a new tracker
                    tracker_id += 1
                    p = mp.Process(target=tracker_start, args=(frame,box,i,tracker_id,inpQ,outQ))
                    p.daemon = True
                    p.start()
        
        [inpQ.put(frame) for inpQ in inpQueues]

        fps = FPS().start()
        
        for outQ in outQueues:
            (success, box, i, tracker_id) = outQ.get()
            text = f"Tracker {tracker_id}: "
            color = [int(c) for c in COLORS[classIDs[i]]]
            if success:
                (x,y,w,h) = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1) 
                info = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i]) 
                cv2.putText(frame, info, (x-5, y-5), FONT, 0.3, color, 1)
                text += "OK"
            else:
                text += "FAILED"

            cv2.putText(frame, text, (W-120,15*(tracker_id)), FONT, 0.4, color,1)

        fps.update()
        fps.stop()
        try:
            fps_ = fps.fps()
            fps_pre = fps_
        except ZeroDivisionError:
            fps_ = fps_pre
        frame_id += 1
        cv2.putText(frame, f"{tracker_type} | YOLO / {N} Frame", (10, 15), FONT, 0.5, (0,0,255), 2)
        cv2.putText(frame, f"FPS: {fps_:.2f}", (10, 30), FONT, 0.5, (0,0,255), 2)
        cv2.putText(frame, f"{frame_id}/{total}", (10, 45), FONT, 0.4, (0,0,255), 2)
        writer.write(frame)
        # cv2.imshow("YOLO", frame)
        if frame_id % N == 0:
            print( f"{frame_id}/{total}")

    vs.release()
    cv2.destroyAllWindows()

# MAIN
#==================================================================
if __name__ == "__main__":

    vs = cv2.VideoCapture(inp)

    # Total number of frames in video stream
    try: 
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT 
        total_frame = int(vs.get(prop)) 
    except:
        total_frame = -1

    vs = cv2.VideoCapture(inp)

    inpQueues = []
    outQueues = []

    main(vs, inpQueues, outQueues, N=N, total=total_frame)
