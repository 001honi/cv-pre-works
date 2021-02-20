import multiprocessing as mp
import numpy as np
import cv2
import imutils
from imutils.video import FPS

tracker_type = "CSRT"
inp = "videos/race-short.mp4"
out = f"videos/race_v3_{tracker_type}_SSD.avi"

# SSD Model Constants 
#===================================================================
prototxt = "model/MobileNetSSD_deploy.prototxt"
model = "model/MobileNetSSD_weights.caffemodel"
conf_thresh = 0.2
net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# OpenCV Constants
#===================================================================
BLUE  = (255,0,0)
GREEN = (0,255,0)
RED   = (0,0,255)
CYAN  = (255,255,0)
MAGENTA = (255,0,255)
COLOR = [GREEN, MAGENTA, CYAN, BLUE]
FONT = cv2.FONT_HERSHEY_SIMPLEX
#===================================================================

def tracker_start(frame, box, id, inpQ, outQ):
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
            if success:
                (x,y,w,h) = [int(v) for v in box]
                box = (x,y,x+w,y+h)
            outQ.put((success, box, id))

def main(vs, inpQueues, outQueues, track=False, writer=False, fps_pre=0.0):
    while True:
        ret, frame = vs.read()
        if not (ret or frame):
            break

        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        if writer is False:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out, fourcc, 30, (W,H), True)

        if track is False:    
            # if inpQueues:
            #     inpQueues = []; outQueues = []    
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            # sort for confidence values and loop
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < conf_thresh:
                    # if true other detections also lower confidence
                    break
                # if label is not appropriate continue
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])
                (x1,y1,x2,y2) = box.astype("int")
                box = (x1,y1,x2-x1,y2-y1)  # x,y,w,h

                # this part same with v2: for multiprocessing
                inpQ = mp.Queue()
                outQ = mp.Queue()
                inpQueues.append(inpQ)
                outQueues.append(outQ)
                # new daemon process for a new tracker
                p = mp.Process(target=tracker_start, args=(frame,box,i,inpQ,outQ))
                p.daemon = True
                p.start()
            track = True
            continue

        [inpQ.put(frame) for inpQ in inpQueues]

        fps = FPS().start()

        for outQ in outQueues:
            (success, box, i) = outQ.get()
            text = f"Tracker {i+1}: "
            ic = 3 if i>3 else i
            if success:
                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), COLOR[ic], 2)
                text += "OK"
            else:
                text += "FAILED"
                # track = False
            
            cv2.putText(frame, text, (W-120,15*(i+1)), FONT, 0.4, COLOR[ic],1)
            
        fps.update()
        fps.stop()
        try:
            fps_ = fps.fps()
            fps_pre = fps_
        except ZeroDivisionError:
            fps_ = fps_pre

        fps_ = f"FPS: {fps_:.2f}"
        cv2.putText(frame, fps_, (10,20), FONT, 0.6, RED, 2)
        writer.write(frame)

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    vs = cv2.VideoCapture(inp)
    trackers = []

    inpQueues = []
    outQueues = []

    main(vs, inpQueues, outQueues)
