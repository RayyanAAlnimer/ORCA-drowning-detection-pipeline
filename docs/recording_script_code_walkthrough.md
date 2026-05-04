# Recording Script: 4-6 Minute Code Walkthrough

Use this for the code section of the recorded demo. The project concept has
already been explained, so this should be a focused implementation walkthrough:
where the pipeline starts, how data moves through the code, and where the alert
is generated.

Target timing: 4 to 6 minutes.

## File Order

Open only these files:

1. `README.md`
2. `scripts/run_yolo_track.py`
3. `pipeline/tracker.py`
4. `pipeline/buffering/track_buffer.py`
5. `pipeline/featurization/pose_features.py`
6. `pipeline/models/tcn.py`
7. `pipeline/decision.py`
8. `pipeline/alert_controller.py`

Mention training briefly from `scripts/run_yolo_track.py` or verbally. Do not
open training files unless asked.

Optional if asked:

- `pipeline/training/trainer.py`
- `pipeline/models/inference.py`
- `scripts/build_dataset.py`
- `pipeline/preprocessing.py`

## 1. Repository Structure

Open `README.md`.

Say:

“For this code walkthrough, I will focus on how the implementation is organised
and how the runtime pipeline connects from input stream to alert output.”

Point to the folder structure.

Say:

“The important folders are `data` for raw and processed datasets, `assets` for
demo media, `models` for YOLO weights and trained checkpoints, `pipeline` for
the core modules, `scripts` for runnable entry points, and `outputs` for logs,
figures, and demo videos.”

Say:

“The runtime flow is: video input, pose tracking, temporal buffering, feature
extraction, TCN classification, decision logic, and alert generation.”

Time target: 25 seconds.

## 2. Runtime Entry Point

Open `scripts/run_yolo_track.py`.

Say:

“This is the video-based runtime entry point. The live camera script follows the
same structure, but this one is easier to show because it uses a fixed video.”

Point to the configuration block.

Say:

“Here we define the YOLO pose model, input video, output path, two-second window,
trained TCN checkpoint, feature size, and decision thresholds.”

Mention quickly:

- `WINDOW_SECONDS = 2.0`
- `FEATURE_DIM = 50`
- `TCN_WEIGHTS_PATH`
- possible threshold `0.60`
- confirmed threshold `0.80`

Say:

“Inside `main`, the script initializes each stage: tracker, buffer, feature
extractor, TCN inferencer, decision manager, and alert controller. The loop then
reads frames, gets tracked poses, builds temporal windows, scores them, and
updates the decision state.”

Time target: 50 seconds.

## 3. Pose Tracking

Open `pipeline/tracker.py`.

Say:

“This module wraps the YOLO pose model. YOLO is not the drowning classifier; it
detects the swimmer, estimates keypoints, gives confidence scores, and maintains
tracking IDs.”

Point to `persist=True`.

Say:

“`persist=True` is the key detail. It keeps the swimmer identity stable across
frames, which lets the next stage build a movement history for the same person.”

Time target: 35 seconds.

## 4. Temporal Buffering

Open `pipeline/buffering/track_buffer.py`.

Say:

“This converts frame-by-frame detections into temporal data. Each track ID gets
its own rolling buffer, so different swimmers do not get mixed together.”

Point to `FrameFeature` and `update()`.

Say:

“Each frame is stored with its frame index, bounding box, confidence, and
keypoints. The buffer only emits a ready sequence once the two-second window is
full.”

Point to age-out.

Say:

“The age-out logic removes stale tracks so old movement history does not affect
future predictions.”

Time target: 40 seconds.

## 5. Feature Extraction

Open `pipeline/featurization/pose_features.py`.

Say:

“This is the bridge between tracking and the model. The model does not receive
raw frames; it receives a compact feature sequence.”

Point to `extract()`.

Say:

“Each buffered window becomes a tensor of shape `(T, 50)`. The 50 features
include normalized keypoints, bounding box information, confidence, velocity,
and motion cues such as speed, acceleration, vertical movement, visibility
ratio, body angle, and stationary duration.”

Point to normalization.

Say:

“Normalization makes the representation less dependent on where the swimmer is
in the frame or how large they appear, so the model focuses more on movement.”

Time target: 60 seconds.

## 6. TCN Classifier

Open `pipeline/models/tcn.py`.

Say:

“The temporal classifier is a TCN. It reads the sequence of pose features over
time and outputs one binary distress score.”

Point to `TemporalBlock`.

Say:

“The model uses one-dimensional convolutions over time, residual connections,
dropout, and increasing dilation. This gives it temporal context while keeping
the model lightweight.”

Point to `TCNBinaryClassifier`.

Say:

“The classifier produces a logit, which is converted into a distress probability
at runtime. That probability is not an alert yet; it is passed into the decision
stage.”

Time target: 45 seconds.

## 7. Decision Logic

Open `pipeline/decision.py`.

Say:

“This stage prevents the model probability from becoming an instant alarm. It
turns probabilities into stable states: normal, possible, and confirmed.”

Point to `TrackDecisionState` and `update_batch()`.

Say:

“Each swimmer has its own decision state. The system smooths probabilities using
an exponential moving average. Above 60 percent becomes possible distress. Above
80 percent, sustained for about one second, becomes confirmed distress.”

Point to cooldown.

Say:

“Cooldown prevents repeated alerts for the same event. This makes the system
more conservative and reduces false alarms from one noisy movement.”

Time target: 60 seconds.

## 8. Alert Controller

Open `pipeline/alert_controller.py`.

Say:

“Once distress is confirmed, the alert controller receives the event and
dispatches it asynchronously, so alert delivery does not block the video loop.”

Point to `handle_alert()` and `_dispatch()`.

Say:

“The alert contains the track ID, probability, and timestamp. It can currently
print to the console, play a sound, send a webhook, or call custom handlers for
hardware-specific outputs.”

Time target: 35 seconds.

## 9. Training Link, Verbally

Say:

“On the training side, labelled videos are processed through the same tracking,
buffering, and feature-extraction stages. The model is trained on the same
`(T, 50)` temporal pose representation that the runtime system produces. The
best checkpoint is saved based on validation F1 and loaded here during runtime.”

Time target: 30 seconds.

## 10. Closing Transition

Say:

“To summarize, the implementation is a staged pipeline. YOLO turns frames into
tracked keypoints. The buffer builds a two-second movement history. Feature
extraction converts that into a 50-dimensional temporal sequence. The TCN
estimates distress probability. Decision logic confirms sustained distress, and
the alert controller produces the final alert.”

Say:

“I have kept this focused on the main implementation path, and I can go deeper
into any file or stage during questions.”

Time target: 25 seconds.

## Leave For Questions

Leave these for Q&A:

- exact formula for all 50 features
- TCN vs LSTM or transformer
- threshold calibration
- training loop details
- multi-swimmer edge cases
- distance and pose-detection limitations
- production validation
