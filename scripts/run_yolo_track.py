from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import queue
import threading
import time

from pipeline.tracker import PoseTracker
from pipeline.buffering import TrackBufferManager
from pipeline.featurization import PoseFeatureExtractor
from pipeline.preprocessing import FramePreprocessor

# Stage 5 (TCN inference)
from pipeline.models import TCNInferencer

# Stage 6 (Decision logic)
from pipeline.decision import DecisionManager, DecisionLevel
from pipeline.alert_controller import AlertController

MODEL_NAME = "yolov8n-pose.pt"
VIDEO_PATH = "assets/videos/real_thrashing_01.mp4"
OUT_PATH = "outputs/yolo_track_out.mp4"

# Lightweight demo/deployment preprocessing before YOLO pose.
PREPROCESSING_ENABLED = True
PREPROCESSING_MODE = "center_zoom_enhance"
PREPROCESSING_ZOOM = 1.35
PREPROCESSING_Y_BIAS = 0.0
SINGLE_SWIMMER_DEMO_TRACK = True

# ------------------------------
# Stage 3 Configuration
# ------------------------------
WINDOW_SECONDS = 2.0      # Temporal window length
AGE_OUT_SECONDS = 1.0     # Track buffer persistence after disappearance

# ------------------------------
# Stage 5 Configuration
# ------------------------------
TCN_WEIGHTS_PATH = "models/checkpoints/mvp-v1/tcn_best.pt"
INFER_EVERY_N_FRAMES = 5        # <-- Run TCN every N frames
MAX_TRACKS_PER_INFER = 2        # <-- Limit how many tracks to score each infer step
FEATURE_DIM = 50                # <-- 42 original features + 8 drowning-specific behavioral features

# ------------------------------
# Stage 6 (decision)
# ------------------------------
DECISION_EMA_ALPHA = 0.30
DECISION_POSSIBLE_TH = 0.60
DECISION_CONFIRM_TH = 0.80
DECISION_HOLD_SEC = 1.0
DECISION_COOLDOWN_SEC = 5.0
DECISION_STALE_SEC = 3.0

# ------------------------------
# Alert controller
# ------------------------------
ALERT_ENABLE_CONSOLE = True
ALERT_ENABLE_SOUND = False
ALERT_WEBHOOK_URL = None

# ------------------------------
# Debug toggles
# ------------------------------
DEBUG_STAGE6 = True
DEBUG_STAGE6_EVERY_N = 1
STAGE6_HEARTBEAT = True


def main():
    print("[TrackRunner] YOLO Pose + Tracking + Stage3 + Stage4 + Stage5 (TCN) + Stage6 (Decision)")
    print("[TrackRunner] VIDEO_PATH:", VIDEO_PATH)

    # --------------------------------------------------
    # Stage 1/2: Video Setup + Tracker Initialization
    # --------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cap.release()

    print(f"[TrackRunner] Video opened OK. fps={fps}, size={w}x{h}")

    out = cv2.VideoWriter(
        OUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    tracker = PoseTracker(MODEL_NAME)
    preprocessor = FramePreprocessor(
        enabled=PREPROCESSING_ENABLED,
        mode=PREPROCESSING_MODE,
        zoom=PREPROCESSING_ZOOM,
        y_bias=PREPROCESSING_Y_BIAS,
    )
    print(
        f"[Preprocess] enabled={PREPROCESSING_ENABLED} mode={PREPROCESSING_MODE} "
        f"zoom={PREPROCESSING_ZOOM} y_bias={PREPROCESSING_Y_BIAS}"
    )
    print(f"[DemoTrack] single_swimmer_demo_track={SINGLE_SWIMMER_DEMO_TRACK}")

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

    alert_controller = AlertController(
        enable_console=ALERT_ENABLE_CONSOLE,
        enable_sound=ALERT_ENABLE_SOUND,
        webhook_url=ALERT_WEBHOOK_URL,
        min_repeat_interval_seconds=DECISION_COOLDOWN_SEC,
    )
    print("[Alert] AlertController ready")

    inference_queue: "queue.Queue[tuple[int, object, int]]" = queue.Queue(maxsize=5)
    result_queue: "queue.Queue[tuple[int, float, int, float]]" = queue.Queue(maxsize=50)
    stop_event = threading.Event()

    def inference_worker():
        while not stop_event.is_set():
            try:
                tid, X, frame_id = inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                p = float(tcn.predict_proba(X))
                ts = time.monotonic()
                if not result_queue.full():
                    result_queue.put((tid, p, frame_id, ts))

                print(f"[Stage5] frame={frame_id:05d} track={tid} p={p:.3f} X={X.shape}", flush=True)
            finally:
                inference_queue.task_done()

    worker_thread = threading.Thread(target=inference_worker, daemon=True)
    worker_thread.start()

    frame_idx = 0
    yielded = 0
    last_log = time.time()
    dbg_event_count = 0

    try:
        # ==================================================
        # Main Processing Loop (File-based Simulation)
        # ==================================================
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = preprocessor.apply(frame)
            r0 = tracker.track_frame(frame)
            yielded += 1

            # --------------------------------------------------
            # Stage 2: Detection + Tracking Output Extraction
            # --------------------------------------------------
            observations = []

            if r0.boxes is not None and len(r0.boxes) > 0:
                ids_t = r0.boxes.id.int() if r0.boxes.id is not None else None
                xyxy_t = r0.boxes.xyxy
                conf_t = r0.boxes.conf if hasattr(r0.boxes, "conf") else None

                kpts_t = None
                if getattr(r0, "keypoints", None) is not None and getattr(r0.keypoints, "data", None) is not None:
                    kpts_t = r0.keypoints.data

                indices = range(len(r0.boxes))
                if SINGLE_SWIMMER_DEMO_TRACK:
                    areas = (xyxy_t[:, 2] - xyxy_t[:, 0]) * (xyxy_t[:, 3] - xyxy_t[:, 1])
                    indices = [int(areas.argmax().item())]

                for i in indices:
                    if not SINGLE_SWIMMER_DEMO_TRACK and ids_t is None:
                        continue
                    tid = 1 if SINGLE_SWIMMER_DEMO_TRACK else int(ids_t[i].item())
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

            # --------------------------------------------------
            # Stage 4 + 5: enqueue windows (non-blocking)
            # --------------------------------------------------
            if ready_sequences and (frame_idx % INFER_EVERY_N_FRAMES == 0):
                for j, (tid, seq) in enumerate(ready_sequences.items()):
                    if j >= MAX_TRACKS_PER_INFER:
                        break
                    batch = extractor.extract(seq)
                    if not inference_queue.full():
                        inference_queue.put((tid, batch.X, frame_idx))

            # --------------------------------------------------
            # Stage 6: decision (consume Stage 5 results)
            # --------------------------------------------------
            probs_batch = {}
            latest_ts = None
            drained = 0

            while True:
                try:
                    tid, p, _, ts = result_queue.get_nowait()
                except queue.Empty:
                    break

                probs_batch[tid] = p
                latest_ts = ts
                drained += 1
                result_queue.task_done()

            if probs_batch:
                events = decision.update_batch(probs_batch, ts=latest_ts)

                if DEBUG_STAGE6 and events:
                    for ev in events:
                        dbg_event_count += 1
                        if (dbg_event_count % DEBUG_STAGE6_EVERY_N) == 0:
                            print(
                                f"[Stage6][DBG] frame={frame_idx:05d} drained={drained} "
                                f"track={ev.track_id} level={ev.level.value} "
                                f"smooth={ev.smooth_prob:.3f} raw={ev.raw_prob:.3f}",
                                flush=True,
                            )

                for ev in events:
                    if ev.level == DecisionLevel.CONFIRMED:
                        alert_controller.handle_alert(
                            track_id=ev.track_id,
                            probability=ev.smooth_prob,
                            timestamp=ev.ts,
                        )

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
            # Visualization (Output Video)
            # --------------------------------------------------
            annotated = r0.plot()

            dec_snapshot = decision.snapshot()
            if r0.boxes is not None and len(r0.boxes) > 0:
                ids_t = r0.boxes.id.int() if r0.boxes.id is not None else None
                xyxy_t = r0.boxes.xyxy
                now_mono = time.monotonic()

                indices = range(len(r0.boxes))
                if SINGLE_SWIMMER_DEMO_TRACK:
                    areas = (xyxy_t[:, 2] - xyxy_t[:, 0]) * (xyxy_t[:, 3] - xyxy_t[:, 1])
                    indices = [int(areas.argmax().item())]

                for i in indices:
                    if not SINGLE_SWIMMER_DEMO_TRACK and ids_t is None:
                        continue
                    tid = 1 if SINGLE_SWIMMER_DEMO_TRACK else int(ids_t[i].item())
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

            now = time.time()
            if now - last_log > 1.0:
                print(
                    f"[Track] frame={frame_idx} active_tracks={len(bufman.get_active_track_ids())} "
                    f"ready_windows={len(ready_sequences)} infer_every={INFER_EVERY_N_FRAMES}",
                    flush=True,
                )

                if STAGE6_HEARTBEAT:
                    snap = decision.snapshot()
                    if snap:
                        items = list(snap.items())[:3]
                        msg = " | ".join(
                            [f"tid={tid} {lvl.value} p={sp:.2f}" for tid, (lvl, sp, _, _) in items]
                        )
                        print(f"[Stage6] {msg}", flush=True)

                last_log = now

            frame_idx += 1

    finally:
        stop_event.set()
        alert_controller.shutdown(timeout=1.0)
        worker_thread.join(timeout=1.0)
        if "cap" in locals():
            cap.release()
        out.release()

    print(f"[Done] Frames processed: {yielded}")
    print(f"[Done] Output written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
