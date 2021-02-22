import cv2

class Tracker():
    ID = 0
    def __init__(self, tracker="KCF"):
        self.tracker_id = Tracker.ID    # Maybe useless
        Tracker.ID += 1
        #-------------------------------------------
        self.tracker = None
        if tracker.upper() == "KCF":
            self.tracker = cv2.TrackerKCF_create()
        elif tracker.upper() == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        #-------------------------------------------
        self.frame_id = None
        self.label_id = None
        self.conf     = None

    def __del__(self):
        # print(f"Tracker {self.tracker_id} is DELETED")
        pass

    def start(self, frame_id, frame, detection):
        self.frame_id = frame_id
        self.label_id = detection[1]
        self.conf     = detection[2]
        self.tracker.init(frame, detection[0])

    def update(self, frame_):
        '''
        Tracks the object in next frame. 
        Returns 
        frame_detection =
                ( frame_id, (box,label_id,conf) )
                or 
                ( frame_id, None )
        '''
        # parsing input
        frame_id, frame = frame_
        # return value
        detection = None

        (success,box) = self.tracker.update(frame)
        if success:
            detection = (box, self.label_id, self.conf)

        return (frame_id, detection)


    