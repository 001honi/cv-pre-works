import numpy as np
import cv2

class YOLO():
    np.random.seed(99)

    configPath  = "model/yolov3.cfg"
    weightsPath = "model/yolov3.weights"
    labelsPath  = "model/coco.txt"
    
    LABELS = open(labelsPath).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS),3), dtype="uint8")

    def __init__(self, conf_thresh=0.5, nms_thresh=0.3):
        self.network   = None
        self.lyr_names = None
        self.conf_thresh = conf_thresh
        self.nms_thresh  = nms_thresh

    def prepare_network(self):
        net = cv2.dnn.readNetFromDarknet(YOLO.configPath, YOLO.weightsPath)
        lyr_names = [net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        self.network = net
        self.lyr_names = lyr_names

    def detect(self,frame,H,W):
        """
        Detects the objects on given one frame.
        Returns 
        detections =
                [(box,label_id,conf), (...)] 
                or 
                None 
        """
        # return value
        detections = None
        # detection result containers
        boxes = [] ; confs = [] ; label_ids = []

        # preprocessing
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        # get network output
        self.network.setInput(blob)
        lyr_outputs = self.network.forward(self.lyr_names)
        
        for output in lyr_outputs:
            for detection in output:
                # select the maximum-likely prediction
                scores = detection[5:]
                label_id = np.argmax(scores)
                conf = scores[label_id]
                if conf < self.conf_thresh:
                    continue        
                # normalize the box coordinates
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # center coordinates to x1,y1
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # append results
                boxes.append([x, y, int(width), int(height)]) 
                confs.append(float(conf)) 
                label_ids.append(label_id)

        # non-maxima-suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thresh, self.nms_thresh)    
        if len(idxs):
            detections = []
            for idx in idxs.flatten():
                box      = tuple(boxes[idx])
                conf     = confs[idx]
                label_id = label_ids[idx]
                detections.append((box,label_id,conf))

        return detections
