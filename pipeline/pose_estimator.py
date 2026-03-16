from ultralytics import YOLO 

class PoseEstimator:
    def __init__(self, model_path=str):
        self.model = YOLO(model_path)

    def infer(self, frame):
        # return Ultralytics Results for this single frame
        return self.model(frame, verbose=False)[0]