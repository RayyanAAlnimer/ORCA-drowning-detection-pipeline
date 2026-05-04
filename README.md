# ORCA Drowning Detection Pipeline

## Overview

ORCA is a real-time drowning distress detection project built around pose-based temporal analysis.

The repository originally began from a fall-detection structure, but the current objective is different: detect drowning-related distress behavior in swimming pools by tracking each person over time, converting their pose history into temporal features, estimating drowning likelihood with a temporal model, stabilizing predictions with decision logic, and triggering structured alerts.

The project now contains two major layers:

- the runtime detection pipeline used for live and video-based inference
- the formal machine learning development pipeline used for dataset building, training, evaluation, calibration, and experimentation

The runtime pipeline has been preserved while the ML development pipeline has been added around it in a modular way.

## Core Runtime Pipeline

The current runtime flow is:

Camera or video input -> YOLO pose detection -> multi-object tracking -> per-track buffering -> pose feature extraction -> TCN inference -> decision logic -> alert controller

The main runtime entry points are:

- `scripts/run_yolo_cam.py` for live webcam inference
- `scripts/run_yolo_track.py` for video-file inference

Presenter guides:

- `docs/code_walkthrough_map.md` gives the recommended code-reading order.
- `docs/recording_script_code_walkthrough.md` is a spoken script for recording a code walkthrough.

## Runtime Stages

### Stage 1 and Stage 2: Pose Detection and Tracking

Files:

- `pipeline/tracker.py`
- `scripts/run_yolo_cam.py`
- `scripts/run_yolo_track.py`

Purpose:

- run YOLO pose estimation on each frame
- assign stable person IDs across frames
- output bounding boxes, keypoints, confidences, and track IDs

### Stage 3: Per-Track Temporal Buffering

Files:

- `pipeline/buffering/track_buffer.py`

Purpose:

- maintain sliding windows of observations for each tracked person
- emit ready temporal sequences only after enough history has accumulated

### Stage 4: Pose Feature Extraction

Files:

- `pipeline/featurization/pose_features.py`

Purpose:

- convert buffered pose history into model-ready feature tensors
- normalize pose relative to body position and scale
- include geometry, confidence, velocity, and drowning-specific behavioral descriptors

### Stage 5: Temporal Inference

Files:

- `pipeline/models/tcn.py`
- `pipeline/models/inference.py`

Purpose:

- run temporal inference on feature sequences
- output drowning-distress probabilities per tracked person

Current note:

- the runtime inference path is implemented
- the MVP-v1 TCN checkpoint is the current primary runtime model for demo inference
- the TCN probability is treated as an input to the decision layer, not as an instant alert by itself

### Stage 6: Decision Logic

Files:

- `pipeline/decision.py`

Purpose:

- smooth noisy probabilities with EMA
- apply hysteresis thresholds
- enforce hold time and cooldown behavior
- emit stable decision states: `normal`, `possible`, `confirmed`

Current demo thresholds:

- `possible_threshold: 0.60`
- `confirmed_threshold: 0.80`
- `ema_alpha: 0.30`
- `hold_time: 1.0` second
- `cooldown: 5.0` seconds

The lower calibrated thresholds previously used in `configs/decision/thresholds.yaml`
made the system more sensitive, but they also increased the chance of confusing
normal swimming or treading with distress. For the MVP-v1 demo, the runtime uses a
more conservative confirmation threshold: a swimmer is first marked as `possible`
when the smoothed distress probability reaches 60%, and is only marked as
`confirmed` when the smoothed probability reaches 80% and remains there for about
one second.

This design intentionally avoids triggering an alert from a single high frame or
short noisy movement. It favors sustained high-confidence distress behavior, which
reduces false positives during non-distress clips.

### Alert Layer

Files:

- `pipeline/alert_controller.py`

Purpose:

- receive confirmed Stage 6 events
- dispatch lightweight alerts asynchronously
- support console alerts, optional sound, optional webhook, and future hardware integrations

## Formal Development Pipeline

A formal ML development pipeline has been added so the project can support structured dataset creation, training, evaluation, calibration, and model comparison without changing the runtime detection architecture.

This new layer supports:

- dataset building from tracked sequences
- training multiple temporal model families
- evaluation on processed datasets
- decision-threshold calibration
- config-driven experimentation
- preservation of older reference training code in a dedicated legacy area

## Project Directory

```text
ORCA-drowning-detection-pipeline/
├── README.md
├── .gitignore
│
├── docs/
│   ├── code_walkthrough_map.md
│   ├── recording_script_code_walkthrough.md
│   └── pipeline_explanation.html
│
├── assets/
│   ├── media/
│   │   └── demo.gif
│   └── videos/
│
├── configs/
│   ├── model/
│   │   ├── tcn.yaml
│   │   ├── lrcn.yaml
│   │   └── tsm.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── epochs_50.yaml
│   └── decision/
│       └── thresholds.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── samples/
│
├── models/
│   ├── pose/
│   │   ├── yolov8n-pose.pt
│   │   └── yolo11n-pose.pt
│   ├── checkpoints/
│   └── exported/
│
├── outputs/
│   └── logs/
│
├── experiments/
│   ├── tcn/
│   ├── lrcn/
│   └── tsm/
│
├── legacy/
│   ├── __init__.py
│   ├── dataset.py
│   ├── loss.py
│   ├── network.py
│   ├── train_tcn_classifier_1.py
│   ├── train_tcn_transformer_1.py
│   ├── train_transformer_classifier_1.py
│   └── stage1_pose-demo.py
│
├── pipeline/
│   ├── __init__.py
│   ├── tracker.py
│   ├── pose_estimator.py
│   ├── visualizer.py
│   ├── decision.py
│   ├── alert_controller.py
│   │
│   ├── buffering/
│   │   ├── __init__.py
│   │   └── track_buffer.py
│   │
│   ├── featurization/
│   │   ├── __init__.py
│   │   └── pose_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── inference.py
│   │   ├── tcn.py
│   │   ├── registry.py
│   │   ├── lrcn.py
│   │   └── tsm.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset_builder.py
│   │   ├── dataloader.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── calibration.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py
│
└── scripts/
    ├── run_yolo_cam.py
    ├── run_yolo_track.py
    ├── run_yolo_only.py
    ├── build_dataset.py
    ├── train_model.py
    ├── evaluate_model.py
    └── calibrate_decision.py
```

## New Development-Pipeline Components

### `configs/`

Stores YAML configuration files.

- `configs/model/tcn.yaml` contains the current TCN model configuration
- `configs/model/lrcn.yaml` defines the future baseline comparison path for LRCN experiments
- `configs/model/tsm.yaml` defines the future advanced comparison path for TSM experiments
- `configs/training/default.yaml` contains baseline training settings
- `configs/decision/thresholds.yaml` contains decision-threshold defaults

### `data/`

Organizes data across development stages.

- `data/raw/` for original inputs
- `data/interim/` for temporary intermediate artifacts
- `data/processed/` for model-ready sequence tensors
- `data/samples/` for smaller curated subsets

### `models/`

Stores trained artifacts.

- `models/pose/` for YOLO pose model weights used before temporal classification
- `models/checkpoints/` for saved checkpoints during training
- `models/exported/` for deployment-ready or converted model outputs later

### `pipeline/utils/`

Utility layer.

- `pipeline/utils/config.py` implements `load_config(path: str) -> dict` using `yaml.safe_load`

### `pipeline/models/registry.py`

Adds model registration so different temporal models can be selected cleanly from scripts.

### `pipeline/models/tcn.py`

Defines the current primary temporal model used by the runtime inference path.

### `pipeline/models/lrcn.py`

Defines the lightweight LRCN baseline for sequence experiments using `(B, T, 50)` feature input.

### `pipeline/models/tsm.py`

Defines the lightweight TSM comparison model using temporal channel shifts over `(B, T, 50)` feature input.

## Model Roles

The repository currently supports three temporal model families:

- `TCN`
  This is the current runtime-oriented temporal model and remains the main reference architecture.
- `LRCN`
  This is the lightweight recurrent baseline model in the development pipeline.
- `TSM`
  This is the lightweight temporal-shift comparison model in the development pipeline.

Both `LRCN` and `TSM` are development-side experimental models. They are available through the training and evaluation scripts but are not wired into the live runtime inference path.

## Current Feature Vector

The current Stage 4 feature extractor outputs a sequence tensor with shape `(T, 50)`.

Per timestep, the 50-dimensional feature vector includes:

- normalized pose coordinates for 17 keypoints
- bounding box geometry
- confidence features
- bbox-center velocity
- speed
- acceleration
- short-term motion variance
- vertical velocity
- area change rate
- visibility ratio
- body angle
- stationary duration

This feature vector is shared by TCN, LRCN, and TSM so model comparisons remain fair.

## Training And Evaluation Workflow

The intended workflow is:

1. collect labeled input videos
2. build processed sequence samples with `scripts/build_dataset.py`
3. train one model family at a time with `scripts/train_model.py`
4. evaluate saved checkpoints with `scripts/evaluate_model.py`
5. compare metrics across models

### 1. Build Processed Samples

Example:

```bash
python scripts/build_dataset.py --video path/to/normal_video.mp4 --label 0
python scripts/build_dataset.py --video path/to/distress_video.mp4 --label 1
```

This uses the existing tracker, buffering, and feature-extraction pipeline to export processed `.pt` samples into `data/processed/`.

### 2. Train A Model

Example for TCN:

```bash
python scripts/train_model.py --model tcn
```

Examples for the comparison models:

```bash
python scripts/train_model.py --model lrcn
python scripts/train_model.py --model tsm
```

Each training run:

- loads processed samples from `data/processed/`
- builds the selected model from its config
- trains using `configs/training/default.yaml`
- saves the best checkpoint to `models/checkpoints/`
- saves training metrics to `outputs/logs/`

### 3. Evaluate A Trained Model

Example for TCN:

```bash
python scripts/evaluate_model.py --model tcn
```

Examples for the comparison models:

```bash
python scripts/evaluate_model.py --model lrcn
python scripts/evaluate_model.py --model tsm
```

Each evaluation run:

- loads the saved checkpoint for the selected model
- evaluates it on processed samples
- writes results to `outputs/logs/<model>_evaluation.json`

### 4. Compare Models Fairly

For a fair comparison, TCN, LRCN, and TSM should all use:

- the same processed dataset
- the same labels
- the same feature vector version
- the same train/validation/test split
- the same evaluation metrics

The most useful metrics for this project are typically:

- recall
- precision
- F1
- later, inference speed and deployment cost

### `pipeline/models/lrcn.py`

Adds `LRCNTemporalClassifier`, a lightweight placeholder for the future baseline comparison model.

### `pipeline/models/tsm.py`

Adds `TSMTemporalClassifier`, a simplified temporal-shift-style experimental model for the more advanced comparison track.

### `pipeline/training/`

Contains reusable training code.

- `dataset_builder.py` builds processed `(T, D)` sequence datasets from tracked sequences
- `dataloader.py` loads saved `.pt` samples for PyTorch training
- `trainer.py` runs training loops and saves best checkpoints
- `losses.py` defines binary cross-entropy loss
- `metrics.py` defines accuracy, precision, recall, and F1

### `pipeline/evaluation/`

Contains reusable evaluation and calibration code.

- `evaluator.py` evaluates trained models and returns metrics plus predictions
- `calibration.py` sweeps thresholds and proposes decision parameters based on evaluation outputs

### `scripts/build_dataset.py`

Builds processed datasets by reusing the runtime tracker, buffer manager, and feature extractor on recorded videos.

### `scripts/train_model.py`

Trains a temporal model with:

- `--model tcn`
- `--model lrcn` for the baseline comparison model
- `--model tsm` for the advanced comparison model

It loads configs, reads processed datasets, and saves checkpoints and metrics.

### `scripts/evaluate_model.py`

Loads a trained checkpoint and evaluates it on processed data.

### `scripts/calibrate_decision.py`

Reads evaluation outputs and generates decision thresholds for the runtime decision layer.

### `legacy/`

Preserves older prototype and training files without deleting them. These are kept for reference and rollback value, while the new development pipeline is organized separately.

## Command Targets

The development pipeline is intended to support the following commands:

```bash
python scripts/build_dataset.py --video path/to/video.mp4 --label 1
python scripts/train_model.py --model tcn
python scripts/train_model.py --model lrcn
python scripts/train_model.py --model tsm
python scripts/evaluate_model.py --model tcn
python scripts/calibrate_decision.py
```

## What Has Been Preserved

The following runtime files remain the protected inference path and should not be treated as training-pipeline code:

- `pipeline/tracker.py`
- `pipeline/buffering/`
- `pipeline/featurization/`
- `pipeline/models/tcn.py`
- `pipeline/models/inference.py`
- `pipeline/decision.py`
- `pipeline/alert_controller.py`
- `scripts/run_yolo_cam.py`
- `scripts/run_yolo_track.py`

The development pipeline extends the repository around these files instead of replacing them.

## Current Status

What is already in place:

- real-time runtime pipeline
- structured alert controller
- formal development-pipeline directories
- config system
- model registry
- baseline support for TCN, LRCN, and TSM experiments
- dataset-building scripts
- training, evaluation, and calibration modules
- preserved legacy branch and legacy files
- 50-dimensional drowning-oriented feature vector shared across all temporal models

What still needs real experimentation:

- building drowning-specific datasets
- training and validating models on real or synthetic drowning sequences
- filling in CNN and TSM architecture/config choices more formally
- calibrating Stage 6 thresholds against real validation results
- integrating trained weights into the runtime pipeline for actual deployment

## Final Summary

This repository now supports both sides of the project:

- a preserved real-time drowning distress detection runtime pipeline
- a formal machine learning development pipeline for building, training, evaluating, and calibrating future models

That structure allows collaborators to work on data, model experimentation, evaluation, and deployment preparation without disrupting the live runtime architecture.
