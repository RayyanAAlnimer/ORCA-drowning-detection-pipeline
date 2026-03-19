import cv2
from ultralytics import YOLO

# 1) Load a YOLO pose model (start with a small one so it runs on Mac CPU)
# You can later swap to YOLOv11 pose weights once you have them.
MODEL_NAME = "yolov8n-pose.pt"

# 2) Input video (use the repo's demo.mp4 first so we can test)
VIDEO_PATH = "demo_from_gif.mp4" # plug a video to test
OUT_PATH = "stage1_pose_out.mp4"

def main():
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 3) Run pose inference on this frame
        results = model(frame, verbose=False)

        # 4) Visualize skeleton + boxes using Ultralytics built-in plot
        annotated = results[0].plot()

        out.write(annotated)

        # 5) Print a lightweight sanity log every ~30 frames
        if frame_idx % 30 == 0:
            r0 = results[0]
            n = 0 if r0.boxes is None else len(r0.boxes)
            print(f"frame={frame_idx:05d} people_detected={n}")

            # If keypoints exist, show shape for the first detected person
            if r0.keypoints is not None and n > 0:
                kpts = r0.keypoints.xy  # shape: (num_people, num_joints, 2)
                print("  keypoints.xy shape:", tuple(kpts.shape))


        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done. Wrote: {OUT_PATH}")

if __name__ == "__main__":
    main()

