"""
YOLOv8 vs Faster R-CNN on the vehicle classes of COCO val2017.

Measures, for each model:
  * accuracy  mAP@0.5 and mAP@0.5:0.95 with the official COCO evaluation (pycocotools),
              plus AP per vehicle type (car, motorcycle, bus, truck)
  * speed     average inference time per image and frames per second (after warm-up)
  * low light the same images darkened artificially, evaluated without and with the
              low-light enhancement used in the app (traffic/preprocess.py)

Both detectors are the standard COCO-pretrained models, so the comparison is fair:
same images, same classes, same evaluation code.

Usage (on a GPU; see notebooks/benchmark_on_kaggle.ipynb):
  python benchmark/compare_models.py --coco-dir /path/to/coco --max-images 500
The COCO folder must contain val2017/ and annotations/instances_val2017.json.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic.config import VEHICLE_CLASSES  # noqa: E402
from traffic.preprocess import enhance_contrast  # noqa: E402

WARMUP = 10
EVAL_CONFIDENCE = 0.001  # standard for mAP: keep low-confidence boxes, the evaluation ranks them


def darken(image: np.ndarray, seed: int) -> np.ndarray:
    """Simulate a night scene: strong gamma darkening plus sensor noise."""
    dark = ((image / 255.0) ** 2.2) * 0.6 * 255
    noise = np.random.default_rng(seed).normal(0, 4, image.shape)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)


def select_images(coco, cat_ids: list[int], max_images: int) -> list[int]:
    """Images that contain at least one vehicle, in a fixed order (so every run uses the same set)."""
    ids = set()
    for cat in cat_ids:
        ids.update(coco.getImgIds(catIds=[cat]))
    return sorted(ids)[:max_images]


def run_model(detector, coco, img_ids, image_dir: Path, name_to_cat: dict, transform=None):
    """Detect on every image. Returns COCO-format results and the inference times (ms)."""
    results, times = [], []
    for i, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        image = cv2.imread(str(image_dir / info["file_name"]))
        if transform:
            image = transform(image, img_id)
        detections, ms = detector.detect(image)
        if i >= WARMUP:
            times.append(ms)
        for d in detections:
            if d.label not in name_to_cat:
                continue
            x1, y1, x2, y2 = d.box
            results.append({"image_id": img_id, "category_id": name_to_cat[d.label],
                            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                            "score": round(d.confidence, 4)})
    return results, times


def evaluate(coco, results, img_ids, cat_ids):
    """Official COCO bbox evaluation restricted to the chosen images and vehicle classes."""
    from pycocotools.cocoeval import COCOeval

    if not results:
        return 0.0, 0.0, {}
    dt = coco.loadRes(results)
    ev = COCOeval(coco, dt, "bbox")
    ev.params.imgIds, ev.params.catIds = img_ids, cat_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    per_class = {}
    for k, cat in enumerate(cat_ids):
        precision = ev.eval["precision"][:, :, k, 0, -1]  # IoU x recall, all areas, max 100 dets
        valid = precision[precision > -1]
        per_class[coco.loadCats(cat)[0]["name"]] = round(float(valid.mean()), 4) if valid.size else 0.0
    return round(float(ev.stats[0]), 4), round(float(ev.stats[1]), 4), per_class


def count_params(detector) -> float:
    model = detector.model.model if hasattr(detector.model, "model") else detector.model
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 1)


def display_name(model: str) -> str:
    return "Faster R-CNN (R50-FPN v2)" if model == "faster_rcnn" else Path(model).stem.replace("yolov8", "YOLOv8")


def plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = report["models"]
    names = [m["name"] for m in models]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(names, [m["map50"] for m in models], color="#1f77b4")
    axes[0].set_title("Accuracy: mAP@0.5 (higher is better)")
    axes[1].bar(names, [m["fps"] for m in models], color="#2ca02c")
    axes[1].set_title("Speed: frames per second (higher is better)")
    if report.get("low_light"):
        ll = report["low_light"]
        x = np.arange(len(ll))
        for i, (key, label) in enumerate([("normal", "normal"), ("dark", "dark"), ("dark_enhanced", "dark + enhancement")]):
            axes[2].bar(x + (i - 1) * 0.27, [r[key] for r in ll], 0.27, label=label)
        axes[2].set_xticks(x, [r["model"] for r in ll])
        axes[2].set_title("Low light: mAP@0.5")
        axes[2].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "faster_rcnn"])
    parser.add_argument("--no-low-light", action="store_true", help="skip the low-light test")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    import torch
    from pycocotools.coco import COCO

    from traffic.detection import make_detector

    coco = COCO(str(args.coco_dir / "annotations" / "instances_val2017.json"))
    image_dir = args.coco_dir / "val2017"
    cat_ids = coco.getCatIds(catNms=list(VEHICLE_CLASSES))
    name_to_cat = {coco.loadCats(c)[0]["name"]: c for c in cat_ids}
    img_ids = select_images(coco, cat_ids, args.max_images)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"{len(img_ids)} images with vehicles · device: {device}")

    report = {"dataset": "COCO val2017, vehicle classes (car, motorcycle, bus, truck)", "images": len(img_ids),
              "device": device, "date": date.today().isoformat(), "models": [], "low_light": []}

    for model in args.models:
        name = display_name(model)
        print(f"\n=== {name} ===")
        detector = make_detector(model, EVAL_CONFIDENCE)
        results, times = run_model(detector, coco, img_ids, image_dir, name_to_cat)
        map50_95, map50, per_class = evaluate(coco, results, img_ids, cat_ids)
        ms = float(np.mean(times))
        report["models"].append({"name": name, "map50": map50, "map50_95": map50_95, "ap_per_class": per_class,
                                 "inference_ms": round(ms, 1), "fps": round(1000 / ms, 1),
                                 "params_m": count_params(detector)})

        if not args.no_low_light:
            dark, _ = run_model(detector, coco, img_ids, image_dir, name_to_cat, transform=lambda im, i: darken(im, i))
            fixed, _ = run_model(detector, coco, img_ids, image_dir, name_to_cat,
                                 transform=lambda im, i: enhance_contrast(darken(im, i), brighten=True))
            report["low_light"].append({"model": name, "normal": map50,
                                        "dark": evaluate(coco, dark, img_ids, cat_ids)[1],
                                        "dark_enhanced": evaluate(coco, fixed, img_ids, cat_ids)[1]})
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2))
    plot(report, args.out / "benchmark.png")

    print("\nModel                        mAP@0.5  mAP@0.5:0.95   FPS   ms/image")
    for m in report["models"]:
        print(f"{m['name']:<28} {m['map50']:.3f}    {m['map50_95']:.3f}      {m['fps']:>6.1f} {m['inference_ms']:>7.1f}")
    for r in report["low_light"]:
        print(f"Low light {r['model']}: normal {r['normal']:.3f} · dark {r['dark']:.3f} · dark+enhanced {r['dark_enhanced']:.3f}")
    print(f"\nSaved {args.out / 'benchmark.json'} and {args.out / 'benchmark.png'}")


if __name__ == "__main__":
    main()
