"""
Vehicle detection with YOLOv8.

The standard YOLOv8 weights (yolov8n/s/m.pt) are trained on the COCO dataset.
COCO has car, motorcycle, bus and truck, but no ambulance, fire truck or police
class. Emergency vehicles can therefore only be detected with custom-trained
weights that include those classes; `supports_emergency` tells the app whether
the loaded model can do it.
"""

import time
from pathlib import Path

# COCO class IDs for the vehicles we count
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Class-name keywords that mean "emergency vehicle" in a custom-trained model
EMERGENCY_KEYWORDS = ("ambulance", "fire truck", "firetruck", "fire_truck", "police", "emergency")

STANDARD_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def is_emergency_class(class_name: str) -> bool:
    name = class_name.lower()
    return any(keyword in name for keyword in EMERGENCY_KEYWORDS)


def list_models(models_dir: str = "models") -> list[str]:
    """Standard YOLOv8 sizes plus any custom .pt files placed in the models/ folder."""
    custom = sorted(str(p) for p in Path(models_dir).glob("*.pt") if p.name not in STANDARD_MODELS)
    return STANDARD_MODELS + custom


class VehicleDetector:
    """YOLOv8 detector that counts vehicles and flags emergency vehicles.

    model_path: a standard name like 'yolov8n.pt' (downloaded automatically on
                first use) or a path to custom weights.
    confidence: minimum confidence for a detection to count (0.0 to 1.0).
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5) -> None:
        from ultralytics import YOLO  # imported here so the other modules work without it

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_names: dict = self.model.names  # {class_id: class_name}
        self.supports_emergency = any(is_emergency_class(n) for n in self.class_names.values())

    def detect(self, frame) -> dict:
        """Run detection on one BGR frame.

        Returns a dict with:
          annotated       frame with boxes and labels drawn
          vehicle_count   number of cars, motorcycles, buses and trucks
          by_type         count per vehicle type, e.g. {"car": 7, "bus": 1}
          emergency       True if an emergency vehicle was found
          emergency_types names of the emergency vehicles found
          inference_ms    model inference time in milliseconds
        """
        start = time.perf_counter()
        result = self.model(frame, conf=self.confidence, verbose=False)[0]
        inference_ms = (time.perf_counter() - start) * 1000

        by_type = {name: 0 for name in VEHICLE_CLASS_IDS.values()}
        emergency_types: list[str] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names.get(cls_id, "")
            if cls_id in VEHICLE_CLASS_IDS and cls_name.lower() == VEHICLE_CLASS_IDS[cls_id]:
                by_type[VEHICLE_CLASS_IDS[cls_id]] += 1
            if is_emergency_class(cls_name):
                label = cls_name.replace("_", " ").title()
                if label not in emergency_types:
                    emergency_types.append(label)

        return {
            "annotated": result.plot(),
            "vehicle_count": sum(by_type.values()),
            "by_type": by_type,
            "emergency": bool(emergency_types),
            "emergency_types": emergency_types,
            "inference_ms": inference_ms,
        }
