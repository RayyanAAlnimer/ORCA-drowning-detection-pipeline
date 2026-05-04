# Models

Use this folder for model artifacts.

- `models/pose/`: YOLO pose weights used by Stage 2.
- `models/checkpoints/`: trained temporal-model checkpoints.
- `models/exported/`: future deployment-ready model exports.

The MVP runtime currently expects:

```text
models/pose/yolov8n-pose.pt
models/checkpoints/mvp-v1/tcn_best.pt
```
