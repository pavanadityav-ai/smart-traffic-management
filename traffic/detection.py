"""
Vehicle detectors with one common interface.

  YoloDetector        Ultralytics YOLOv8 (n, s, m or custom weights). Fast; used live.
  FasterRCNNDetector  torchvision Faster R-CNN (ResNet-50 FPN v2). Slower, two-stage;
                      used to compare accuracy and speed with YOLOv8.

Both standard models are trained on COCO, which has car, motorcycle, bus and
truck but no ambulance, fire truck or police class. Emergency vehicles are
detected only by custom YOLOv8 weights trained on an emergency-vehicle dataset
(see notebooks/train_emergency_kaggle_dataset.ipynb); `supports_emergency` says
whether the loaded model can do it.

ultralytics and torch are imported inside the classes, so the rest of the
project (and the tests) work without them.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from traffic.config import EMERGENCY_KEYWORDS, VEHICLE_CLASSES

STANDARD_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "faster_rcnn"]


@dataclass
class Detection:
    box: tuple          # (x1, y1, x2, y2) in pixels
    label: str          # "car", "motorcycle", "bus", "truck" or an emergency class name
    confidence: float
    emergency: bool = False


def is_emergency_class(name: str) -> bool:
    name = name.lower().replace("_", " ")
    return any(k in name for k in EMERGENCY_KEYWORDS) and name != "fire hydrant"


def keep_detection(name: str) -> bool:
    """Only vehicles and emergency vehicles are kept; people, traffic lights, etc. are ignored."""
    return name.lower() in VEHICLE_CLASSES or is_emergency_class(name)


def to_detection(box, name: str, confidence: float) -> Detection:
    emergency = is_emergency_class(name)
    label = name.lower().replace("_", " ") if emergency else name.lower()
    return Detection(tuple(float(v) for v in box), label, float(confidence), emergency)


def custom_models(models_dir: str = "models") -> list[str]:
    """Custom .pt weights placed in models/ (for example an emergency-vehicle model)."""
    return sorted(str(p) for p in Path(models_dir).glob("*.pt") if p.name not in STANDARD_MODELS)


def list_models(models_dir: str = "models") -> list[str]:
    """Standard models plus any custom .pt files placed in models/."""
    return STANDARD_MODELS + custom_models(models_dir)


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.4) -> None:
        from ultralytics import YOLO

        self.name = Path(weights).stem
        self.model = YOLO(weights)
        self.confidence = confidence
        self.class_names = {int(k): v for k, v in self.model.names.items()}
        self.supports_emergency = any(is_emergency_class(n) for n in self.class_names.values())

    def detect(self, frame) -> tuple[list[Detection], float]:
        start = time.perf_counter()
        result = self.model(frame, conf=self.confidence, verbose=False)[0]
        ms = (time.perf_counter() - start) * 1000
        detections = []
        for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            name = self.class_names.get(int(cls), "")
            if keep_detection(name):
                detections.append(to_detection(box, name, conf))
        return detections, ms


class FasterRCNNDetector:
    def __init__(self, confidence: float = 0.4) -> None:
        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=confidence).to(self.device).eval()
        self.categories = weights.meta["categories"]  # index = COCO category id
        self.name = "faster_rcnn"
        self.confidence = confidence
        self.supports_emergency = False

    def detect(self, frame) -> tuple[list[Detection], float]:
        torch = self.torch
        rgb = frame[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            out = self.model([tensor])[0]
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        detections = []
        for box, label, score in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
            name = self.categories[label]
            if score >= self.confidence and keep_detection(name):
                detections.append(to_detection(box, name, score))
        return detections, ms


class CombinedDetector:
    """Main detector (counts all vehicles) + emergency detector (ambulance, fire truck, police).

    A model trained only on emergency vehicles cannot count ordinary cars, so both
    run on each frame. A main-model box that overlaps an emergency box is dropped,
    so the ambulance is not counted twice (once as "truck", once as "ambulance").
    """

    def __init__(self, main, emergency) -> None:
        self.main, self.emergency = main, emergency
        self.name = f"{main.name}+{emergency.name}"
        self.supports_emergency = emergency.supports_emergency

    def detect(self, frame) -> tuple[list[Detection], float]:
        from traffic.tracking import iou

        main_dets, ms_main = self.main.detect(frame)
        em_dets, ms_em = self.emergency.detect(frame)
        em_dets = [d for d in em_dets if d.emergency]
        kept = [d for d in main_dets if all(iou(d.box, e.box) < 0.5 for e in em_dets)]
        return kept + em_dets, ms_main + ms_em


def make_detector(model: str, confidence: float, emergency_model: str | None = None):
    main = FasterRCNNDetector(confidence) if model == "faster_rcnn" else YoloDetector(model, confidence)
    if emergency_model:
        return CombinedDetector(main, YoloDetector(emergency_model, confidence))
    return main
