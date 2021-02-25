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

    # def __del__(self):
    #     print(f"Tracker {self.tracker_id} is DELETED")
    # #     pass

    def start(self, f, frame, detection):
        self.frame_id = f
        self.label_id = detection[1]
        self.conf     = detection[2]
        self.tracker.init(frame, detection[0])

    def update(self, frame):
        '''
        Tracks the object in next frame. 
        Returns 
        detection =
                (box,label_id,conf) 
                or 
                None 
        '''
        # return value
        detection = None

        (success,box) = self.tracker.update(frame)
        if success:
            detection = (box, self.label_id, self.conf)

        return detection


    