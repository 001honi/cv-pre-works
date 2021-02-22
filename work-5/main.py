from video import Video
from yolo import YOLO
from tracker import Tracker
from tqdm import tqdm
import time

path = "videos/sample.mp4"
tracker_type = "CSRT"
N = 30
title = f"YOLO_{N}_{tracker_type}"

if __name__ == "__main__":

    video = Video(path)
    video.read_frames()
    (H,W) = video.shape

    yolo = YOLO(conf_thresh=0.65, nms_thresh=0.5)
    yolo.prepare_network()

    start_time = time.time()

    for f in tqdm(range(video.total_frame)):
        if f % N == 0:
            # yolo detection
            frame_detections = yolo.detect(video.frames_inp[f],H,W)
            # start trackers
            frame_id, detections = frame_detections
            if detections:
                trackers = []
                frame = video.frames_inp[frame_id][1]  # pick the related frame
                for detection in detections:
                    trackers.append(Tracker(tracker_type))
                    trackers[-1].start(frame_id, frame, detection)
        else:
            frame_detections = []
            for tracker in trackers:
                frame_detection = tracker.update(video.frames_inp[f])
                frame_id, detection = frame_detection
                if detection:
                    frame_detections.append(detection)

            frame_detections = (f, frame_detections)

        fps = (f+1)/(time.time()-start_time)
        video.put_frame(frame_detections, fps, YOLO.LABELS, YOLO.COLORS)

    video.write(f"videos/sample_{title}.avi")

    elapsed_time = time.time() - start_time
    print(title)
    print(f"Elapsed time: {elapsed_time:.3f} secs")
    print(f"Average FPS: {video.total_frame/elapsed_time:.2f}")
