# Code Presentation Script And Directions

Use this document while recording your VS Code walkthrough. The goal is to show
the examiner that you understand the project structure, the runtime pipeline,
the training pipeline, and why each module exists.

## Recording Setup

Before recording:

1. Open the project root in VS Code.
2. Open the Explorer sidebar so the folder structure is visible.
3. Keep this document open in a split pane.
4. Do not explain every line. Explain the purpose of each file and pause only on
   the most important code blocks.

Suggested recording order:

1. `README.md`
2. `docs/code_walkthrough_map.md`
3. `scripts/run_yolo_track.py`
4. `pipeline/preprocessing.py`
5. `pipeline/tracker.py`
6. `pipeline/buffering/track_buffer.py`
7. `pipeline/featurization/pose_features.py`
8. `pipeline/models/tcn.py`
9. `pipeline/models/inference.py`
10. `pipeline/decision.py`
11. `pipeline/alert_controller.py`
12. `scripts/build_dataset.py`
13. `pipeline/training/trainer.py`
14. `scripts/evaluate_model.py`
15. `outputs/README.md`

## 1. Project Structure

Open:

`README.md`

Say:

“I will now walk through the software implementation of the drowning detection
system. The repository is organised around the full development pipeline. The
`assets` folder stores demonstration and evaluation media. The `data` folder is
where raw and processed datasets are placed. The `models` folder contains pose
weights and trained checkpoints. The `pipeline` folder contains the core
implementation, `scripts` contains runnable entry points, and `outputs` contains
generated logs, figures, diagnostics, and demo videos.”

Then open:

`docs/code_walkthrough_map.md`

Say:

“This file gives the route through the code. I will follow the runtime path
first, from video input to alert generation, and then briefly show the training
and evaluation path.”

## 2. Runtime Entry Point

Open:

`scripts/run_yolo_track.py`

Say:

“I am starting with the video-based runtime script because it shows the complete
pipeline in one place. The live camera version follows the same structure in
`scripts/run_yolo_cam.py`.”

Point to the configuration section.

Say:

“At the top, the script defines the YOLO pose model, input video, output path,
preprocessing options, the temporal window size, the TCN checkpoint, and the
decision thresholds. The important settings are a two-second movement window, a
50-dimensional feature vector, and conservative decision thresholds of 60 percent
for possible distress and 80 percent for confirmed distress.”

Mention:

- `WINDOW_SECONDS = 2.0`
- `FEATURE_DIM = 50`
- `TCN_WEIGHTS_PATH`
- `DECISION_POSSIBLE_TH = 0.60`
- `DECISION_CONFIRM_TH = 0.80`
- `DECISION_HOLD_SEC = 1.0`

Say:

“This file coordinates the pipeline, but the actual logic is separated into
modules so each stage has a clear responsibility.”

## 3. Stage 1: Input And Preprocessing

Open:

`pipeline/preprocessing.py`

Say:

“The first stage is optional preprocessing. This improves visibility before pose
detection. The system can apply contrast enhancement and crop-zooming to make the
swimmer clearer under glare, distance, or low contrast.”

Point to `apply()`.

Say:

“The key design choice is that preprocessing returns a normal BGR frame with the
same output size. This means the YOLO pose detector does not need a separate
interface or special input format.”

Point to `_enhance()` and `_crop_zoom()`.

Say:

“Contrast enhancement improves local detail, and crop-zooming makes the swimmer
occupy more pixels while preserving the expected frame shape.”

## 4. Stage 2: Pose Detection And Tracking

Open:

`pipeline/tracker.py`

Say:

“This module wraps the YOLO pose model. YOLO is not detecting drowning directly.
Its role is to detect the swimmer, estimate body keypoints, produce bounding
boxes and confidence values, and maintain tracking IDs.”

Point to `persist=True`.

Say:

“The `persist=True` option is important because it keeps swimmer identities
stable across frames. Without tracking, each frame would be independent. With
tracking, we can build a movement history for each swimmer.”

Return to:

`scripts/run_yolo_track.py`

Point to the observation extraction block.

Say:

“Here, the YOLO result is converted into a simple observation format. Each
observation contains the track ID, frame index, bounding box, confidence, and
keypoints. This is the handoff from raw computer vision output into the temporal
pipeline.”

## 5. Stage 3: Temporal Buffering

Open:

`pipeline/buffering/track_buffer.py`

Say:

“This stage stores a rolling movement history for every tracked swimmer. Each
track ID has its own buffer, so two swimmers do not get mixed together.”

Point to `update()`.

Say:

“The buffer only emits a ready sequence when enough frames have been collected.
In the MVP setup, this is around 60 frames, or two seconds at 30 FPS. This is
important because drowning is not obvious from a single frame; it is a pattern of
movement over time.”

Point to the age-out logic.

Say:

“The age-out logic handles short disappearances. If a swimmer disappears briefly,
the buffer remains alive. If the track is missing for too long, the old history
is removed so stale data does not contaminate future predictions.”

## 6. Stage 4: Feature Extraction

Open:

`pipeline/featurization/pose_features.py`

Say:

“This is one of the most important parts of the pipeline. It converts the
two-second pose history into numerical features that the temporal model can
understand.”

Point to `extract()`.

Say:

“Each sequence becomes a tensor with shape `(T, 50)`. Each frame is represented
by 50 features. These include normalized keypoint positions, bounding box
geometry, confidence values, velocity, and behaviour-related cues such as speed,
acceleration, motion variance, vertical movement, area change, visibility ratio,
body angle, and stationary duration.”

Point to the normalization code.

Say:

“The normalization step is important because it makes the model less dependent on
where the swimmer appears in the frame or how large they appear due to camera
distance. This helps the model focus more on movement pattern than raw pixel
position.”

## 7. Stage 5: TCN Model And Inference

Open:

`pipeline/models/tcn.py`

Say:

“The temporal classifier is a TCN, or Temporal Convolutional Network. It uses
one-dimensional convolutions over time to learn movement patterns from the pose
sequence.”

Point to `TCNBinaryClassifier`.

Say:

“The model input is not raw video. It is a sequence of pose features. The model
outputs a logit, which is later converted into a distress probability.”

Then open:

`pipeline/models/inference.py`

Say:

“This file wraps the trained TCN checkpoint for runtime use. Stage 4 gives the
features as `(time, features)`, and the inferencer transposes them into the
channels-first format expected by the TCN.”

Point to `torch.sigmoid`.

Say:

“The TCN only produces a probability. It does not trigger an alert by itself.
That probability is passed into the decision layer.”

## 8. Background Inference Queue

Return to:

`scripts/run_yolo_track.py`

Point to `inference_worker()` and the queues.

Say:

“TCN inference runs in a background thread. This keeps the video loop responsive.
When a two-second sequence is ready, it is converted into features and placed in
the inference queue. The worker returns a probability for that track.”

Say:

“This separation matters for real-time performance because video capture,
tracking, and buffering should continue even while the classifier is scoring a
sequence.”

## 9. Stage 6: Decision Logic

Open:

`pipeline/decision.py`

Say:

“This stage turns noisy model probabilities into stable behavioural states. The
states are `normal`, `possible`, and `confirmed`.”

Point to `DecisionManager`.

Say:

“A raw probability spike is not enough to trigger an alert. The system first
smooths probabilities using an exponential moving average. If the smoothed value
passes 60 percent, the track becomes possible distress. If it stays above 80
percent for about one second, the track becomes confirmed distress.”

Point to cooldown.

Say:

“Cooldown prevents repeated alerts for the same event. The aim is to respond to
sustained distress behaviour, not momentary noise or one unusual movement.”

## 10. Alert Generation

Open:

`pipeline/alert_controller.py`

Say:

“Once distress is confirmed, the alert controller handles notification. It runs
asynchronously, so alert delivery does not block video processing.”

Point to `handle_alert()` and `_dispatch()`.

Say:

“Currently, the alert can be printed to the console, played as a sound, sent to a
webhook, or extended later for hardware integration. There is also duplicate
suppression as a second guard against repeated alerts.”

## 11. Dataset Building

Open:

`scripts/build_dataset.py`

Say:

“Now I will briefly show how the model was trained. The dataset builder takes
labeled videos and turns them into processed temporal sequence samples.”

Point to the saved sequence block.

Say:

“Each saved sample mirrors the runtime inference window. It is not a raw frame;
it is a temporal pose sequence with a label.”

Then open:

`pipeline/training/dataset_builder.py`

Say:

“This module saves each sample as a `.pt` file containing the feature tensor
`X`, the label `y`, the track ID, and metadata. This ensures the training data
uses the same representation as the runtime system.”

## 12. Training And Evaluation

Open:

`scripts/train_model.py`

Say:

“This script loads the model and training configuration, builds the selected
model, creates the data loaders, and calls the trainer.”

Open:

`pipeline/training/trainer.py`

Say:

“The trainer implements the epoch loop. For each epoch, it trains the model,
evaluates on the training and validation sets, records metrics, and saves the
best checkpoint based on validation F1.”

Say:

“F1 is useful here because this task cares about both false positives and missed
distress cases.”

Open:

`scripts/evaluate_model.py`

Say:

“Evaluation loads a trained checkpoint and writes metrics such as accuracy,
precision, recall, F1, mAP, predictions, and targets into the outputs folder.”

## 13. Inputs, Models, And Outputs

Open:

`data/README.md`

Say:

“Raw labelled videos go into `data/raw`, processed temporal tensors go into
`data/processed`, and smaller test subsets can go into `data/samples`.”

Open:

`models/README.md`

Say:

“YOLO pose weights are stored in `models/pose`, while trained temporal model
checkpoints are stored in `models/checkpoints`. The MVP runtime loads the TCN
checkpoint from here.”

Open:

`outputs/README.md`

Say:

“Generated logs, metrics, report figures, diagnostics, and demo videos are stored
under `outputs`. This keeps generated artifacts separate from implementation
code.”

## 14. Closing Summary

Say:

“To summarize, the implementation is a staged pipeline. The input stream provides
frames. Preprocessing improves visibility. YOLO pose detection converts frames
into swimmer keypoints. Tracking gives each swimmer a persistent identity.
Temporal buffering collects two seconds of movement. Feature extraction converts
that movement into a 50-dimensional sequence. The TCN estimates distress
probability. Decision logic smooths and confirms sustained distress, and the
alert controller generates the final alert.”

Say:

“The main strength of this design is that each stage has a clear responsibility.
The main limitation is that the system depends on pose quality. Distance, glare,
occlusion, or missing keypoints can reduce the reliability of the temporal
classifier.”

## 5-Minute Short Version

If the recording needs to be shorter, use this simplified route:

1. `README.md`
2. `scripts/run_yolo_track.py`
3. `pipeline/tracker.py`
4. `pipeline/buffering/track_buffer.py`
5. `pipeline/featurization/pose_features.py`
6. `pipeline/models/tcn.py`
7. `pipeline/decision.py`
8. `pipeline/alert_controller.py`
9. `pipeline/training/trainer.py`
10. `outputs/README.md`

Short closing line:

“The code is organised so the runtime pipeline and the development pipeline are
separate but consistent. The model is trained on the same pose-sequence
representation that the live system produces, and the alert is only triggered
after the model probability is smoothed and confirmed over time.”
