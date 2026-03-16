import cv2

from pipeline.tracker import PoseTracker
from pipeline.buffering import TrackBufferManager
from pipeline.featurization import PoseFeatureExtractor

# Stage 5 (TCN inference)
from pipeline.models import TCNInferencer

# Stage 6 (Decision logic)
from pipeline.decision import DecisionManager, DecisionLevel

MODEL_NAME = "yolov8n-pose.pt"
VIDEO_PATH = "assets/videos/input.mp4"
OUT_PATH = "outputs/yolo_track_out.mp4"

# ------------------------------
# Stage 3 Configuration
# ------------------------------
WINDOW_SECONDS = 2.0      # Temporal window length
AGE_OUT_SECONDS = 1.0     # Track buffer persistence after disappearance

# ------------------------------
# Stage 5 Configuration
# ------------------------------
TCN_WEIGHTS_PATH = None         # <-- Set to a .pt checkpoint once you have trained weights
INFER_EVERY_N_FRAMES = 5        # <-- Run TCN every N frames
MAX_TRACKS_PER_INFER = 2        # <-- Limit how many tracks to score each infer step
FEATURE_DIM = 42                # <-- update if Stage 4 features change

# ------------------------------
# Stage 6 (decision)
# ------------------------------
DECISION_EMA_ALPHA = 0.30
DECISION_POSSIBLE_TH = 0.60
DECISION_CONFIRM_TH = 0.80
DECISION_HOLD_SEC = 1.0
DECISION_COOLDOWN_SEC = 5.0
DECISION_STALE_SEC = 3.0


def main():
    print("[Stage2] Starting YOLO+Tracking runner...")
    print("[Stage2] VIDEO_PATH:", VIDEO_PATH)

    # --------------------------------------------------
    # Stage 1/2: Video Setup + Tracker Initialization
    # --------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"[Stage2] Video opened OK. fps={fps}, size={w}x{h}")

    out = cv2.VideoWriter(
        OUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    tracker = PoseTracker(MODEL_NAME)

    # --------------------------------------------------
    # Stage 3: Per-Track Temporal Buffer Initialization
    # --------------------------------------------------
    window_size_frames = max(1, int(round(WINDOW_SECONDS * float(fps))))
    age_out_frames = max(0, int(round(AGE_OUT_SECONDS * float(fps))))

    bufman = TrackBufferManager(
        window_size_frames=window_size_frames,
        age_out_frames=age_out_frames
    )

    print(f"[Stage3] Window={window_size_frames} frames | Age-out={age_out_frames} frames")

    # --------------------------------------------------
    # Stage 4: Feature Extractor Initialization
    # --------------------------------------------------
    extractor = PoseFeatureExtractor(
        num_keypoints=17,
        use_velocity=True,
        include_bbox=True,
        include_conf=True,
        frame_size=(w, h)
    )

    print("[Stage4] PoseFeatureExtractor initialized")

    # --------------------------------------------------
    # Stage 5: TCN inference
    # --------------------------------------------------
    tcn = TCNInferencer(feature_dim=FEATURE_DIM, weights_path=TCN_WEIGHTS_PATH)
    print(f"[Stage5] TCNInferencer ready (device={tcn.device}, weights={TCN_WEIGHTS_PATH})")

    # --------------------------------------------------
    # Stage 6: Decision manager
    # --------------------------------------------------
    decision = DecisionManager(
        ema_alpha=DECISION_EMA_ALPHA,
        possible_th=DECISION_POSSIBLE_TH,
        confirm_th=DECISION_CONFIRM_TH,
        hold_seconds=DECISION_HOLD_SEC,
        cooldown_seconds=DECISION_COOLDOWN_SEC,
        stale_track_seconds=DECISION_STALE_SEC,
    )
    print("[Stage6] DecisionManager ready")

    frame_idx = 0
    yielded = 0

    # ==================================================
    # Main Processing Loop (Live Simulation)
    # ==================================================
    for r0 in tracker.track_video(VIDEO_PATH):
        yielded += 1

        # --------------------------------------------------
        # Stage 2: Detection + Tracking Output Extraction
        # --------------------------------------------------
        observations = []

        if r0.boxes is not None and r0.boxes.id is not None and len(r0.boxes) > 0:
            ids_t = r0.boxes.id.int()
            xyxy_t = r0.boxes.xyxy
            conf_t = r0.boxes.conf if hasattr(r0.boxes, "conf") else None

            kpts_t = None
            if getattr(r0, "keypoints", None) is not None and getattr(r0.keypoints, "data", None) is not None:
                kpts_t = r0.keypoints.data

            for i in range(len(r0.boxes)):
                tid = int(ids_t[i].item())
                bbox_xyxy = xyxy_t[i].tolist()
                conf = float(conf_t[i].item()) if conf_t is not None else 1.0

                kpts_xyc = None
                if kpts_t is not None and i < kpts_t.shape[0]:
                    kpts_xyc = kpts_t[i].tolist()

                observations.append({
                    "track_id": tid,
                    "frame_idx": frame_idx,
                    "bbox_xyxy": bbox_xyxy,
                    "conf": conf,
                    "kpts_xyc": kpts_xyc,
                })

        # --------------------------------------------------
        # Stage 3: Update Temporal Buffers
        # --------------------------------------------------
        ready_sequences = bufman.update(observations, frame_idx)

        if frame_idx % 30 == 0:
            n = 0 if r0.boxes is None else len(r0.boxes)
            ids = []
            if r0.boxes is not None and r0.boxes.id is not None:
                ids = r0.boxes.id.int().tolist()

            print(f"[Stage2] frame={frame_idx:05d} people={n} ids={ids}")
            print(
                f"[Stage3] active_tracks={len(bufman.get_active_track_ids())} "
                f"ready_windows={len(ready_sequences)}"
            )

        # --------------------------------------------------
        # Stage 4: Convert Ready Windows → Model Tensors
        # Stage 5: Run TCN every N frames
        # Stage 6: Decision update
        # --------------------------------------------------
        if ready_sequences and (frame_idx % INFER_EVERY_N_FRAMES == 0):
            probs_batch = {}
            latest_ts = None

            for j, (tid, seq) in enumerate(ready_sequences.items()):
                if j >= MAX_TRACKS_PER_INFER:
                    break

                batch = extractor.extract(seq)
                p = float(tcn.predict_proba(batch.X))
                latest_ts = cv2.getTickCount() / cv2.getTickFrequency()

                probs_batch[tid] = p

                if frame_idx % 60 == 0:
                    print(
                        f"[Stage5] frame={frame_idx:05d} track={tid} "
                        f"p={p:.3f} X={batch.X.shape} mask={batch.mask.shape}"
                    )

            if probs_batch:
                events = decision.update_batch(probs_batch, ts=latest_ts)
                for ev in events:
                    if ev.level == DecisionLevel.CONFIRMED:
                        # Replace with your actual alert controller later
                        print(
                            f"[ALERT] frame={frame_idx:05d} track={ev.track_id} "
                            f"smooth={ev.smooth_prob:.3f} raw={ev.raw_prob:.3f}",
                            flush=True
                        )

        # --------------------------------------------------
        # Visualization (Output Video)
        # --------------------------------------------------
        annotated = r0.plot()

        # Optional: overlay decision state
        dec_snapshot = decision.snapshot()
        if r0.boxes is not None and r0.boxes.id is not None and len(r0.boxes) > 0:
            ids_t = r0.boxes.id.int()
            xyxy_t = r0.boxes.xyxy
            now_mono = cv2.getTickCount() / cv2.getTickFrequency()

            for i in range(len(r0.boxes)):
                tid = int(ids_t[i].item())
                x1, y1, x2, y2 = [int(v) for v in xyxy_t[i].tolist()]
                lvl, sp, _, cooldown_until = dec_snapshot.get(
                    tid, (DecisionLevel.NORMAL, 0.0, 0.0, 0.0)
                )
                cd_left = max(0.0, cooldown_until - now_mono)
                text = f"id={tid} p={sp:.2f} {lvl.value}"
                if cd_left > 0:
                    text += f" cd={cd_left:.1f}s"

                cv2.putText(
                    annotated,
                    text,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        out.write(annotated)

        frame_idx += 1

    out.release()

    print(f"[Done] Frames processed: {yielded}")
    print(f"[Done] Output written to: {OUT_PATH}")


if __name__ == "__main__":
    main()