# Code Walkthrough Map

Use this file as the route through the repository when explaining the project.

## Runtime Path

The runtime system takes frames from a camera or video file and turns them into a
stable alert decision.

1. `scripts/run_yolo_cam.py`
   - Live camera entry point.
   - Shows the full runtime loop from camera input to alert handling.

2. `scripts/run_yolo_track.py`
   - Video-file version of the same pipeline.
   - Easier to present because it uses a fixed input video and output path.

3. `pipeline/preprocessing.py`
   - Optional frame enhancement and zoom.
   - Keeps the output frame shape unchanged so YOLO can run normally.

4. `pipeline/tracker.py`
   - Wraps YOLO pose tracking.
   - Converts raw frames into boxes, keypoints, confidences, and persistent track IDs.

5. `pipeline/buffering/track_buffer.py`
   - Maintains a rolling two-second history for each tracked swimmer.
   - Emits a sequence only when enough frames exist.

6. `pipeline/featurization/pose_features.py`
   - Converts each buffered sequence into the model input shape `(T, 50)`.
   - Normalizes pose and adds motion descriptors.

7. `pipeline/models/tcn.py`
   - Defines the temporal classifier.
   - Uses causal temporal convolution blocks and a binary output head.

8. `pipeline/models/inference.py`
   - Loads the trained TCN checkpoint.
   - Converts a `(T, 50)` sequence into one distress probability.

9. `pipeline/decision.py`
   - Converts noisy probabilities into `normal`, `possible`, and `confirmed`.
   - Uses EMA smoothing, thresholds, hold time, and cooldown.

10. `pipeline/alert_controller.py`
    - Sends confirmed alerts asynchronously.
    - Supports console, sound, webhook, and future hardware handlers.

## Development Path

The development pipeline creates datasets and trains/evaluates models.

1. `scripts/build_dataset.py`
   - Converts labeled videos into `.pt` temporal sequence samples.

2. `pipeline/training/dataset_builder.py`
   - Saves each extracted temporal sequence with its label.

3. `pipeline/training/dataloader.py`
   - Loads saved `.pt` samples for training.

4. `scripts/train_model.py`
   - Loads config, builds the selected temporal model, and trains it.

5. `pipeline/training/trainer.py`
   - Implements the epoch loop, metrics, and best-checkpoint saving.

6. `scripts/evaluate_model.py`
   - Evaluates a trained checkpoint on processed data.

7. `pipeline/evaluation/evaluator.py`
   - Computes final predictions, targets, and metrics.

## Configuration Files

- `configs/model/tcn.yaml`: TCN architecture settings.
- `configs/training/default.yaml`: training epochs, batch size, learning rate, and device.
- `configs/decision/thresholds.yaml`: MVP decision thresholds.

## Input And Model Locations

- `data/raw/`: raw training videos, grouped by class.
- `assets/videos/`: curated demonstration and real-position test videos.
- `models/pose/`: YOLO pose weights used by Stage 2.
- `models/checkpoints/`: trained temporal model checkpoints, including `mvp-v1`.

## Generated Artifacts

Generated files are intentionally ignored by git:

- `data/`: raw and processed datasets.
- `outputs/`: logs, figures, diagnostics, and demo videos.
- `models/checkpoints/`: trained checkpoints.
- Python and plotting caches.
