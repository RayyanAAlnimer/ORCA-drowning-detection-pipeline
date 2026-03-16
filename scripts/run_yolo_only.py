import cv2
from pipeline.pose_estimator import PoseEstimator
from pipeline.visualizer import draw_results

MODEL_NAME = "yolov8n-pose.pt" # we will test with this small model first, you can swap to other YOLO models later
VIDEO_PATH = "assets/videos/input.mp4" 
OUT_PATH = "outputs/yolo_only_out.mp4"

def main():
    pose = PoseEstimator(MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: 
            break

        r0 = pose.infer(frame)
        annotated = draw_results(r0)
        out.write(annotated)

        if frame_idx % 30 == 0:
            n = 0 if r0.boxes is None else len(r0.boxes)
            print(f"frame={frame_idx:05d} people_detected={n}")
            
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done. Wrote: {OUT_PATH}")

if __name__ == "__main__":
    main()


# To run this script as a module: 
#   python -m scripts.run_yolo_only