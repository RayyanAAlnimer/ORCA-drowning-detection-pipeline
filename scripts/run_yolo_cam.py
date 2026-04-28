from __future__ import annotations

import sys
import signal
from pathlib import Path
import cv2
import time
import threading
import queue

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Terminal raw key reading (macOS/Linux)
import termios
import tty
import select

from pipeline.tracker import PoseTracker
from pipeline.buffering import TrackBufferManager
from pipeline.featurization import PoseFeatureExtractor
from pipeline.preprocessing import FramePreprocessor

# Stage 5 (TCN inference)
from pipeline.models import TCNInferencer

# Stage 6 (Decision logic)
from pipeline.decision import DecisionManager, DecisionLevel
from pipeline.alert_controller import AlertController

# =========================
# USER CONFIG (edit here)
# =========================
MODEL_NAME = "yolov8n-pose.pt"   # <-- Change YOLO pose model here (e.g., yolov11n-pose.pt)
CAMERA_INDEX = 0                # <-- 0 = default webcam, try 1 if you have multiple
DISPLAY_SCALE = 1.0             # <-- e.g., 0.75 to reduce load (helps FPS)
WRITE_OUTPUT = False            # <-- True to save annotated video
OUT_PATH = "outputs/yolo_cam_out.mp4"

# Lightweight preprocessing before YOLO pose. Keep this enabled for the demo camera path.
PREPROCESSING_ENABLED = True
PREPROCESSING_MODE = "center_zoom_enhance"
PREPROCESSING_ZOOM = 1.35
PREPROCESSING_Y_BIAS = 0.0
SINGLE_SWIMMER_DEMO_TRACK = True

# Stage 3 (buffering)
WINDOW_SECONDS = 2.0
AGE_OUT_SECONDS = 1.0

# Stage 4 (features)
USE_VELOCITY = True
INCLUDE_BBOX = True
INCLUDE_CONF = True

# Stage 5 (TCN)
TCN_WEIGHTS_PATH = "models/checkpoints/mvp-v1/tcn_best.pt"
INFER_EVERY_N_FRAMES = 5        # <-- Run TCN every N frames to keep realtime smooth
MAX_TRACKS_PER_INFER = 2        # <-- Limit how many tracks to score each infer step

# Stage 6 (decision) -- thresholds are low now for testing, adjust later based on TCN performance
DECISION_EMA_ALPHA = 0.30
DECISION_POSSIBLE_TH = 0.10
DECISION_CONFIRM_TH = 0.20
DECISION_HOLD_SEC = 0.2
DECISION_COOLDOWN_SEC = 5.0
DECISION_STALE_SEC = 3.0

# =========================
# DEBUG TOGGLES
# =========================
DEBUG_STAGE6 = True          # <-- turn off later
DEBUG_STAGE6_EVERY_N = 1     # print every decision event (set to 5/10 if too spammy)

# Print a 1Hz Stage6 heartbeat alongside the [Live] log (recommended for demos)
STAGE6_HEARTBEAT = True
# =========================

# Alert controller
ALERT_ENABLE_CONSOLE = True
ALERT_ENABLE_SOUND = False
ALERT_WEBHOOK_URL = None


def main():
    print("[CamRunner] Live webcam YOLO Pose + Tracking + Stage3 + Stage4 + Stage5 (TCN) + Stage6 (Decision)")
    print("[CamRunner] Quit from window: press 'q' | Quit from terminal: press 'q' | Ctrl+C supported")

    # ONE shared stop event for everything
    stop_event = threading.Event()

    def request_shutdown(reason: str):
        if not stop_event.is_set():
            print(f"\n[Shutdown] {reason}", flush=True)
            stop_event.set()

    # Ctrl+C handler
    def signal_handler(sig, frame):
        request_shutdown("Ctrl+C received")

    signal.signal(signal.SIGINT, signal_handler)

    # ----------------------------
    # Camera init
    # ----------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"[Cam] fps≈{fps}, size={w}x{h}")

    out = None
    if WRITE_OUTPUT:
        out = cv2.VideoWriter(
            OUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        print(f"[Cam] Recording enabled -> {OUT_PATH}")

    # ----------------------------
    # Stage 2: tracker
    # ----------------------------
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

    # ----------------------------
    # Stage 3: buffering
    # ----------------------------
    window_size_frames = max(1, int(round(WINDOW_SECONDS * float(fps))))
    age_out_frames = max(0, int(round(AGE_OUT_SECONDS * float(fps))))
    bufman = TrackBufferManager(window_size_frames=window_size_frames, age_out_frames=age_out_frames)
    print(f"[Stage3] window={window_size_frames} frames | age_out={age_out_frames} frames")

    # ----------------------------
    # Stage 4: features
    # ----------------------------
    extractor = PoseFeatureExtractor(
        num_keypoints=17,
        use_velocity=USE_VELOCITY,
        include_bbox=INCLUDE_BBOX,
        include_conf=INCLUDE_CONF,
        frame_size=(w, h),
    )
    print("[Stage4] extractor ready")

    # ----------------------------
    # Stage 5: TCN inference (background)
    # ----------------------------
    FEATURE_DIM = 50  # <-- 42 original features + 8 drowning-specific behavioral features
    tcn = TCNInferencer(feature_dim=FEATURE_DIM, weights_path=TCN_WEIGHTS_PATH)
    print(f"[Stage5] TCNInferencer ready (device={tcn.device}, weights={TCN_WEIGHTS_PATH})")

    # ----------------------------
    # Stage 6: Decision manager
    # ----------------------------
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
    if DEBUG_STAGE6:
        print(
            "[Stage6][DBG] enabled "
            f"(alpha={DECISION_EMA_ALPHA}, possible={DECISION_POSSIBLE_TH}, confirm={DECISION_CONFIRM_TH}, "
            f"hold={DECISION_HOLD_SEC}s, cooldown={DECISION_COOLDOWN_SEC}s)",
            flush=True
        )

    inference_queue: "queue.Queue[tuple[int, object, int]]" = queue.Queue(maxsize=5)

    # Worker -> main thread results (tid, prob, frame_id, ts_monotonic)
    result_queue: "queue.Queue[tuple[int, float, int, float]]" = queue.Queue(maxsize=50)

    def inference_worker():
        while not stop_event.is_set():
            try:
                tid, X, frame_id = inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                p = float(tcn.predict_proba(X))
                ts = time.monotonic()

                # Send probability to main thread for Stage 6
                if not result_queue.full():
                    result_queue.put((tid, p, frame_id, ts))

                print(f"[Stage5] frame={frame_id:05d} track={tid} p={p:.3f} X={X.shape}", flush=True)
            finally:
                inference_queue.task_done()

    worker_thread = threading.Thread(target=inference_worker, daemon=True)
    worker_thread.start()

    # ----------------------------
    # Terminal 'q' listener (no Enter required)
    # ----------------------------
    fd = sys.stdin.fileno()
    old_term_settings = termios.tcgetattr(fd)

    def keyboard_listener():
        # Put terminal into cbreak mode: read single chars immediately
        tty.setcbreak(fd)
        try:
            while not stop_event.is_set():
                # Non-blocking check
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    if ch and ch.lower() == "q":
                        request_shutdown("Terminal 'q' pressed")
                        break
        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term_settings)

    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    # ----------------------------
    # Main loop
    # ----------------------------
    frame_idx = 0
    last_log = time.time()
    window_name = "Live YOLO Pose + Tracking (press q to quit)"

    # Debug counters (Stage 6)
    dbg_event_count = 0

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                request_shutdown("Camera frame read failed")
                break

            if DISPLAY_SCALE != 1.0:
                frame = cv2.resize(frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))

            frame = preprocessor.apply(frame)

            # Stage 2: pose + tracking
            r0 = tracker.track_frame(frame)

            # Stage 3: observations
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
                    kpts_xyc = kpts_t[i].tolist() if kpts_t is not None and i < kpts_t.shape[0] else None

                    observations.append({
                        "track_id": tid,
                        "frame_idx": frame_idx,
                        "bbox_xyxy": bbox_xyxy,
                        "conf": conf,
                        "kpts_xyc": kpts_xyc,
                    })

            ready_sequences = bufman.update(observations, frame_idx)

            # Stage 4 + 5: enqueue windows (non-blocking)
            if ready_sequences and (frame_idx % INFER_EVERY_N_FRAMES == 0):
                for j, (tid, seq) in enumerate(ready_sequences.items()):
                    if j >= MAX_TRACKS_PER_INFER:
                        break
                    batch = extractor.extract(seq)
                    if not inference_queue.full():
                        inference_queue.put((tid, batch.X, frame_idx))

            # ----------------------------
            # Stage 6: decision (consume Stage 5 results)
            # ----------------------------
            probs_batch = {}
            latest_ts = None
            drained = 0

            while True:
                try:
                    tid, p, p_frame_id, ts = result_queue.get_nowait()
                except queue.Empty:
                    break

                probs_batch[tid] = p
                latest_ts = ts
                drained += 1
                result_queue.task_done()

            if probs_batch:
                events = decision.update_batch(probs_batch, ts=latest_ts)

                # ==========================================================
                # DEBUG STAGE 6 (SAFE): prints ONLY when Stage 6 emits events
                # ==========================================================
                if DEBUG_STAGE6 and events:
                    for ev in events:
                        dbg_event_count += 1
                        if (dbg_event_count % DEBUG_STAGE6_EVERY_N) == 0:
                            print(
                                f"[Stage6][DBG] frame={frame_idx:05d} drained={drained} "
                                f"track={ev.track_id} level={ev.level.value} "
                                f"smooth={ev.smooth_prob:.3f} raw={ev.raw_prob:.3f}",
                                flush=True
                            )
                # ==========================================================

                for ev in events:
                    if ev.level == DecisionLevel.CONFIRMED:
                        alert_controller.handle_alert(
                            track_id=ev.track_id,
                            probability=ev.smooth_prob,
                            timestamp=ev.ts,
                        )

            # Display
            annotated = r0.plot()

            # Optional debug overlay for decision state
            dec_snapshot = decision.snapshot()
            if r0.boxes is not None and len(r0.boxes) > 0:
                ids_t = r0.boxes.id.int() if r0.boxes.id is not None else None
                xyxy_t = r0.boxes.xyxy
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
                    cd_left = max(0.0, cooldown_until - time.monotonic())
                    text = f"{lvl.value} p={sp:.2f}"
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

            cv2.imshow(window_name, annotated)

            if out is not None:
                out.write(annotated)

            # Window-key quit (works when loop is responsive)
            key = cv2.waitKey(1)
            if key in (ord("q"), ord("Q")):
                request_shutdown("Window 'q' pressed")

            # Optional: exit if window is closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                request_shutdown("Window closed")

            # Periodic log (1Hz)
            now = time.time()
            if now - last_log > 1.0:
                print(
                    f"[Live] frame={frame_idx} active_tracks={len(bufman.get_active_track_ids())} "
                    f"ready_windows={len(ready_sequences)} infer_every={INFER_EVERY_N_FRAMES}",
                    flush=True
                )

                # Stage 6 heartbeat (also 1Hz) — avoids spamming every frame
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
        # Clean shutdown
        stop_event.set()
        alert_controller.shutdown(timeout=1.0)
        worker_thread.join(timeout=1.0)
        listener_thread.join(timeout=1.0)

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        # Ensure terminal settings are restored even if we crash before listener finishes
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term_settings)
        except Exception:
            pass

        print("[CamRunner] Clean shutdown complete.")


if __name__ == "__main__":
    main()
