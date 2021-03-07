from video import Video
from mask_rcnn import MaskRCNN
from tqdm import tqdm
import time

path = "videos/sample.mp4"
title = f"MASK_RCNN"

video = Video(path,resize_w=500)
video.read_frames()
(H,W) = video.shape

mrcnn = MaskRCNN()
mrcnn.prepare_network(mode="inference")

if __name__ == "__main__":

    start_time = time.time()

    print("[INFO] Detections are collecting..")
    for f in tqdm(range(video.total_frame)):
        # yolo detection
        detections = mrcnn.detect(video.frames_inp[f],verbose=0)
        if detections:
            fps = (f+1)/(time.time()-start_time)
            video.put_frame(f,detections,LABELS=MaskRCNN.LABELS,fps=fps)

    video.write(f"videos/sample_{title}.avi")

    elapsed_time = time.time() - start_time
    print(title)
    print(f"Elapsed time: {elapsed_time:.3f} secs")
    print(f"Average FPS: {video.total_frame/elapsed_time:.2f}")
