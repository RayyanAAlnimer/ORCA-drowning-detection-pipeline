# ORCA Drowning Detection Pipeline

## 1. Project Context

This project is a real-time drowning distress detection system built around pose-based temporal analysis. Although the repository originally grew out of a fall-detection codebase, the current goal is different: to detect drowning-related distress behavior in swimming pools by analyzing how a person's pose and motion evolve over time.

The main idea is that drowning distress is not a single-frame event. A single image may show a swimmer in an unusual posture, but that alone is not enough to classify distress reliably. Because of that, the pipeline tracks each person over time, stores short windows of observations, converts those windows into model-ready features, and then applies a temporal classifier before making a final decision.

At a high level, the pipeline works as follows:

Camera or video input -> YOLO pose detection -> multi-object tracking -> sliding-window buffering -> feature extraction -> TCN inference -> decision logic -> alert controller

The live webcam pipeline runs through `scripts/run_yolo_cam.py`, and the file-based video pipeline runs through `scripts/run_yolo_track.py`.

## 2. Why the System Is Structured in Stages

The code is organized as a staged pipeline because each part has a clear and different responsibility.

- Early stages are responsible for seeing people and keeping their identities stable over time.
- Middle stages are responsible for transforming raw detections into temporal model input.
- Later stages are responsible for classification, stabilizing predictions, and triggering alerts.

This design keeps the system modular, easier to debug, and easier to extend later for trained drowning models, hardware alarms, and deployment-specific integrations.

## 3. Stage 1 and Stage 2: Pose Detection and Multi-Object Tracking

### Purpose

Stage 1 and Stage 2 are effectively combined through the YOLO pose tracking pipeline. Their job is to process each incoming frame and identify all visible people while also assigning each detected person a stable track ID.

### Main Code

- `pipeline/tracker.py`
- `scripts/run_yolo_cam.py`
- `scripts/run_yolo_track.py`

### Input

A single image frame from either:

- a live webcam in `run_yolo_cam.py`, or
- a video file in `run_yolo_track.py`

### Processing

The `PoseTracker` class wraps Ultralytics YOLO pose tracking. For each frame, YOLO produces:

- bounding boxes
- keypoints
- detection confidences
- persistent tracking IDs

The use of `persist=True` is important because it allows the same person to keep the same identity across frames.

### Output

For each detected person in a frame, the pipeline extracts:

- `track_id`
- `bbox_xyxy`
- `conf`
- `kpts_xyc`
- `frame_idx`

Those observations are then passed to the buffering stage.

## 4. Stage 3: Sliding-Window Buffering

### Purpose

Stage 3 is responsible for temporal memory. A temporal model cannot classify drowning distress from one frame alone, so the system needs a rolling history for each tracked person.

### Main Code

- `pipeline/buffering/track_buffer.py`

### Input

A list of observations for the current frame. Each observation contains:

- track ID
- frame index
- bounding box
- confidence
- pose keypoints

### Processing

`TrackBufferManager` keeps a separate rolling buffer for each active `track_id`. Each incoming observation is converted into a lightweight `FrameFeature` object and appended to the correct per-track deque.

The buffer manager also:

- drops stale tracks after a configured age-out period
- emits only full windows once enough frames have accumulated

### Output

A dictionary of ready temporal sequences:

- key: `track_id`
- value: list of buffered `FrameFeature` objects

Only tracks with a fully populated window are forwarded to Stage 4.

## 5. Stage 4: Pose Feature Extraction

### Purpose

Stage 4 converts raw buffered pose observations into fixed-size numerical features suitable for temporal inference.

### Main Code

- `pipeline/featurization/pose_features.py`

### Input

A buffered sequence for one person, containing pose, box, and confidence information across multiple timesteps.

### Processing

`PoseFeatureExtractor` performs several steps:

- safely reads pose keypoints even if some are missing or malformed
- converts the bounding box to center-width-height form
- optionally normalizes by frame size
- chooses a person-centered coordinate system using hips and shoulders when available
- normalizes pose coordinates relative to that center and scale
- appends optional auxiliary features such as:
  - bounding box geometry
  - detection confidence
  - mean keypoint confidence
  - simple velocity from bbox center displacement

This makes the features more robust to where the person appears in the image and emphasizes body motion and posture instead of absolute image position.

### Output

A `FeatureBatch` object containing:

- `X`: a feature matrix of shape `(T, D)`
- `mask`: a timestep mask
- `meta`: debug metadata

This `X` tensor is the direct input to Stage 5.

## 6. Stage 5: Temporal Inference with the TCN

### Purpose

Stage 5 estimates how likely the observed sequence corresponds to drowning distress.

### Main Code

- `pipeline/models/tcn.py`
- `pipeline/models/inference.py`

### Input

A feature tensor `X` of shape `(T, D)` representing the motion history of one tracked person.

### Processing

The TCN inference wrapper converts the features into the shape expected by the temporal model and runs a forward pass through a causal Temporal Convolutional Network.

The causal design matters because the model only uses current and past timesteps. That makes it suitable for real-time use.

In both `run_yolo_cam.py` and the updated `run_yolo_track.py`, Stage 5 runs through a background worker thread. The main loop pushes feature windows into an inference queue, and the worker returns probabilities through a result queue.

This keeps inference from blocking the rest of the pipeline.

### Output

A probability value per tracked person representing the estimated likelihood of drowning distress.

### Current State

The Stage 5 path is fully implemented, but the model has not yet been trained for the drowning task. That means the architecture is ready, but the probabilities are still placeholders until trained weights are integrated.

## 7. Stage 6: Decision Logic

### Purpose

Raw model probabilities should not directly trigger alarms because they can fluctuate. Stage 6 stabilizes the model output and turns it into a more reliable behavioral state.

### Main Code

- `pipeline/decision.py`

### Input

A batch of per-track probabilities produced by Stage 5.

### Processing

`DecisionManager` applies several pieces of real-time logic:

- exponential moving average smoothing
- hysteresis thresholds
- hold-time requirements
- cooldown handling
- stale-track cleanup

The decision states are:

- `normal`
- `possible`
- `confirmed`

A person only reaches `confirmed` if the model score stays high enough for long enough according to the configured thresholds and hold time.

### Output

A list of `DecisionEvent` objects describing state transitions per track.

This is the point where the system moves from "model output" to "actionable event."

## 8. Alert Controller

### Purpose

The alert controller replaces the older placeholder alert print with a structured, extensible alert system.

### Main Code

- `pipeline/alert_controller.py`

### Input

Confirmed Stage 6 events, passed as:

- `track_id`
- `probability`
- `timestamp`

### Processing

The `AlertController` accepts alerts through `handle_alert(track_id, probability, timestamp)`.

To keep the main pipeline lightweight, `handle_alert()` does not perform heavy work directly. Instead, it:

- checks duplicate suppression timing
- queues the alert for asynchronous handling

A background worker thread then dispatches the alert.

### Current Alert Outputs

The controller currently supports:

- console alert logging
- optional sound alarm
- optional webhook trigger
- future custom handlers for embedded or hardware integrations

### Why This Design Matters

This keeps alerting modular and deployment-friendly. The same detection pipeline can later support:

- a local siren
- a remote API notification
- an Arduino-triggered alarm
- a Jetson or GPIO-based hardware response

without redesigning the detection logic.

## 9. How the End-to-End Flow Works

The full system flow is:

1. A frame enters the system from a camera or a video file.
2. YOLO pose tracking detects people and assigns track IDs.
3. Each tracked person's observations are added to a rolling buffer.
4. Full temporal windows are converted into model-ready features.
5. The TCN scores those windows and outputs drowning probabilities.
6. The decision layer smooths and stabilizes those probabilities.
7. Confirmed events are passed to the alert controller.
8. Alerts are dispatched while the video output continues to be annotated and written.

This means the system is not just a classifier. It is a real-time perception and response pipeline.

## 10. Live Pipeline vs File-Based Pipeline

### `scripts/run_yolo_cam.py`

This is the live webcam runner.

It is designed for real-time streaming and now includes:

- Stage 1 to Stage 6
- background Stage 5 inference
- decision overlays
- alert controller integration

### `scripts/run_yolo_track.py`

This is the file-based video runner.

It now mirrors the webcam runner much more closely and includes:

- the same stage-based pipeline
- queue-based Stage 5 inference
- Stage 6 decision logic
- alert controller integration
- annotated output video writing

This runner will be especially useful once trained drowning weights are available because it will let the system process recorded videos and generate both annotated outputs and structured alerts.

## 11. Current System Status

The pipeline is structurally complete and functioning end-to-end.

What already exists:

- YOLO pose detection
- tracking
- buffering
- feature extraction
- TCN inference wrapper
- decision logic
- alert controller
- live and file-based runners

What is not complete yet:

- trained drowning-specific TCN weights
- fully validated drowning behavior dataset integration
- production hardware alert deployment

So at this stage, the architecture is in place, but model intelligence still depends on the upcoming training process.

## 12. Future Additions

Based on the current development direction, the next major additions are expected to be:

### 12.1 Training the TCN on Drowning Data

The most important next step is to train the temporal model on drowning distress sequences, including synthetic data already generated for that purpose.

That will allow Stage 5 to produce meaningful drowning probabilities instead of placeholders.

### 12.2 Integrating Trained Weights

Once training is complete, trained model weights will be connected directly into the existing inference path.

That should require minimal architectural change because the Stage 5 interface is already prepared for loading checkpoints.

### 12.3 Expanding the Alert System

The alert controller is intentionally designed for future extensions such as:

- webhook-based external notifications
- audio alarms
- GPIO triggers
- Arduino integrations
- Jetson-based deployment hooks

### 12.4 Deployment and Threshold Tuning

After the model is trained, the decision parameters will likely need tuning based on validation behavior. This includes:

- smoothing strength
- possible and confirmed thresholds
- hold time
- cooldown behavior

That tuning step will be important for balancing sensitivity and false alarms.

## 13. Final Summary

In summary, this project is a real-time drowning distress detection pipeline built on top of pose estimation and temporal analysis.

Its strength comes from combining multiple layers of processing:

- frame-level human pose estimation
- identity-aware tracking
- temporal buffering
- normalized feature extraction
- sequence classification with a TCN
- stabilized decision logic
- modular alert dispatch

The codebase is already organized in a way that supports real-time deployment without needing to redesign the architecture later. The next phase is to train the drowning model, integrate real weights, and extend deployment-facing components such as alerting and hardware triggers.

That makes the current system a strong structural foundation for a deployable drowning distress detection solution.
