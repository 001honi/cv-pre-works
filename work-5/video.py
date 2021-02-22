import numpy as np
import cv2
import imutils


class Video():
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    BLUE = (255,0,0) 
    GREEN= (0,255,0)
    RED  = (0,0,255)

    def __init__(self, path):
        self.path = path
        self.frames_inp = []
        self.frames_out = []
        self.total_frame = None
        self.shape = (None,None) # H,W

    def read_frames(self):
        """
        stores all the frames in the given video source in 
            self.frames_inp (list) as [(frame_id, frame),(...)]
        """
        try: 
            source = cv2.VideoCapture(self.path)
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT 
            self.total_frame = int(source.get(prop)) 
        except:
            print("Error in Path or Frame Count")

        for index in range(self.total_frame):
            ret, frame = source.read()
            if not (ret or frame):
                print("Error in Frame Read")
                break
            frame = imutils.resize(frame, width=500)
            if not self.shape[0]:
                self.shape = frame.shape[:2]
            frame = (index,frame)
            self.frames_inp.append(frame)

    def put_frame(self, frame_detections, fps, LABELS, COLORS):
        """
        applies the detections on related frame; appends to
            self.frames_out (list) as [(frame_id, frame),(...)]
        """
        frame_id, detections = frame_detections
        frame = self.frames_inp[frame_id][1]  # pick the related frame
        if detections:
            for detection in detections:
                (box, label_id, conf) = detection

                label = "{}: {:.2f}".format(LABELS[label_id], conf) 
                C = tuple([int(i) for i in COLORS[label_id]]) # label specific color

                cv2.rectangle(frame, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), C, 2) 
                cv2.putText(frame, label, (box[0]-5, box[1]-5), Video.FONT, 0.3, C, 1)
        # add fps and frame_id
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 15), Video.FONT, 0.5, Video.RED, 1)
        cv2.putText(frame, f"{frame_id}/{self.total_frame}", (10, 30), Video.FONT, 0.3, Video.RED, 1)

        self.frames_out.append((frame_id, frame))

    def write(self, path="out.avi"):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, 30, (self.shape[1],self.shape[0]), True)
        for frame in self.frames_out:
            frame = frame[1]
            writer.write(frame)