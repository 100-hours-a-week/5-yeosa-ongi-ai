from app.model.yoloface.face_detector import YoloDetector

device = "cpu"  # or "cuda" if available
_yolo_detector = None


def load_yolo_detector(target_size: int = 320, min_face: int = 60):
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YoloDetector(
            target_size=target_size, device=device, min_face=min_face
        )
    return _yolo_detector
