import cv2
import imutils

from mrcnn import visualize

class Video():
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    BLUE = (255,0,0) 
    GREEN= (0,255,0)
    RED  = (0,0,255)

    def __init__(self, path, resize_w=500):
        self.path = path
        self.resize_w = resize_w
        self.frames_inp = []
        self.frames_out = []
        self.total_frame = None
        self.shape = (None,None) # H,W

    def read_frames(self):
        """
        stores all the frames in the given video source in 
            self.frames_inp (list) as [frame0, frame1, ...]
        where frame# is numpy array
        """
        try: 
            source = cv2.VideoCapture(self.path)
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT 
            self.total_frame = int(source.get(prop)) 
        except:
            print("Error in Path or Frame Count")

        for i in range(self.total_frame):
            ret, frame = source.read()
            if not (ret or frame):
                print("Error in Frame Read")
                break
            frame = imutils.resize(frame, width=self.resize_w)
            if not self.shape[0]:
                self.shape = frame.shape[:2]

            self.frames_inp.append(frame)
        
        print("[INFO] Video Import Completed")

    def put_frame(self,f,r,LABELS,fps=None):
        """
        applies the detections on related frame; appends to
            self.frames_out (list) as [frame0, frame1, ...]
        """
        frame = self.frames_inp[f]  # pick the related frame

        # Visualize results
        for i in range(0, r["rois"].shape[0]):
            mask = r["masks"][:, :, i]
            frame = visualize.apply_mask(frame, mask,(1.0, 0.0, 0.0), alpha=0.5)
            frame = visualize.draw_box(frame, r["rois"][i], (1.0, 0.0, 0.0))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for i in range(0, len(r["scores"])):
            (startY, startX, endY, end) = r["rois"][i]
            classID = r["class_ids"][i]
            label = LABELS[classID]
            score = r["scores"][i]

            text = "{}: {:.4f}".format(label, score)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y), Video.FONT, 0.8, Video.GREEN, 2)
        
         # add fps and f
        if fps:
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), Video.FONT, 0.5, Video.RED, 1)
        cv2.putText(frame, f"{f}/{self.total_frame}", (10, 15), Video.FONT, 0.3, Video.RED, 1)

        self.frames_out.append(frame)

    def write(self, path="out.avi"):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, 30, (self.shape[1],self.shape[0]), True)
        for frame in self.frames_out:
            writer.write(frame)
        print("[INFO] Video Export Completed")
        