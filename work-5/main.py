from video import Video
from yolo import YOLO
from tracker import Tracker
from tqdm import tqdm
import time

path = "videos/sample.mp4"
tracker_type = "CSRT"
N = 30
title = f"YOLO_{N}_{tracker_type}"

video = Video(path)
video.read_frames()
(H,W) = video.shape

yolo = YOLO(conf_thresh=0.65, nms_thresh=0.5)
yolo.prepare_network()

if __name__ == "__main__":

    start_time = time.time()

    print("[INFO] Detections are collecting..")
    for f in tqdm(range(video.total_frame)):
        if f % N == 0:
            # yolo detection
            detections = yolo.detect(video.frames_inp[f],H,W)
            if detections:
                trackers = []
                frame = video.frames_inp[f]  # pick the related frame
                for detection in detections:
                    if detection:
                        trackers.append(Tracker(tracker_type))
                        trackers[-1].start(f, frame, detection)
        else:
            detections = []
            for tracker in trackers:
                detection = tracker.update(video.frames_inp[f])
                if detection:
                    detections.append(detection)

            frame_detections = (f,detections)

        fps = (f+1)/(time.time()-start_time)
        video.put_frame(f,detections,LABELS=YOLO.LABELS,COLORS=YOLO.COLORS,fps=fps)

    video.write(f"videos/sample_{title}.avi")

    elapsed_time = time.time() - start_time
    print(title)
    print(f"Elapsed time: {elapsed_time:.3f} secs")
    print(f"Average FPS: {video.total_frame/elapsed_time:.2f}")
