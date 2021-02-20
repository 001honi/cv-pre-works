import multiprocessing as mp
import cv2
import imutils
from imutils.video import FPS

tracker_type = "CSRT"
n_tracker = 4
inp = "videos/race-short.mp4"
out = f"videos/race_v2_{tracker_type}_{n_tracker}.avi"

#=================================================
BLUE  = (255,0,0)
GREEN = (0,255,0)
RED   = (0,0,255)
CYAN  = (255,255,0)
MAGENTA = (255,0,255)
COLOR = [GREEN, MAGENTA, CYAN, BLUE]
FONT = cv2.FONT_HERSHEY_SIMPLEX
#=================================================

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
            for i in range(n_tracker):
                box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
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
            if success:
                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), COLOR[i], 2)
                text += "OK"
            else:
                text += "FAILED"
            
            cv2.putText(frame, text, (W-120,15*(i+1)), FONT, 0.4, COLOR[i],1)
            
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
