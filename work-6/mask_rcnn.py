import cv2

import mrcnn.model as modellib
from model.coco import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNN():
    weightsPath = "model/coco/mask_rcnn_coco.h5"
    labelsPath  = "model/coco/coco.txt"

    LABELS = open(labelsPath).read().strip().split("\n")

    def __init__(self):
        self.network = None

    def prepare_network(self, mode="inference"):
        config = InferenceConfig()
        config.display()
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference",model_dir="./logs",config=config)
        # Load weights trained on MS-COCO
        model.load_weights(MaskRCNN.weightsPath, by_name=True)
        self.network = model

    def detect(self,frame,verbose=1):
        """
        Detects the objects on given one frame.
        Returns 
            default 'results' in Matterport implementation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.network.detect([frame],verbose=verbose)[0]