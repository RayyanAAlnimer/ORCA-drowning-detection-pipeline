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
- optionally include box geometry, confidence, and simple velocity features

### Stage 5: Temporal Inference

Files:

- `pipeline/models/tcn.py`
- `pipeline/models/inference.py`

Purpose:

- run temporal inference on feature sequences
- output drowning-distress probabilities per tracked person

Current note:

- the runtime inference path is implemented
- trained drowning-specific weights are not yet fully integrated
- current probabilities should still be treated as placeholders until training is completed

### Stage 6: Decision Logic

Files:

- `pipeline/decision.py`

Purpose:

- smooth noisy probabilities with EMA
- apply hysteresis thresholds
- enforce hold time and cooldown behavior
- emit stable decision states: `normal`, `possible`, `confirmed`

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
├── yolov8n-pose.pt
├── demo.gif
│
├── docs/
│   └── pipeline_explanation.html
│
├── configs/
│   ├── model/
│   │   ├── tcn.yaml
│   │   ├── lrcn.yaml
│   │   └── tsm.yaml
│   ├── training/
│   │   └── default.yaml
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

- `models/checkpoints/` for saved checkpoints during training
- `models/exported/` for deployment-ready or converted model outputs later

### `pipeline/utils/`

Utility layer.

- `pipeline/utils/config.py` implements `load_config(path: str) -> dict` using `yaml.safe_load`

### `pipeline/models/registry.py`

Adds model registration so different temporal models can be selected cleanly from scripts.

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
- baseline support for TCN, CNN, and TSM experiments
- dataset-building scripts
- training, evaluation, and calibration modules
- preserved legacy branch and legacy files

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
