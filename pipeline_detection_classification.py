# pipeline_detection_classification.py
# Combined YOLOv8 detection + MobileNetV2 classifier pipeline
# (safe, ready-to-run; adjust paths if your files live elsewhere)

import os
from pathlib import Path
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# ---- CONFIG - update if your files are in a different place ----
PROJECT_ROOT = Path("C:/Users/beena/Aerial_Project").resolve()

# YOLO run folder name used during training (example: bird_drone_yolov8_light)
YOLO_RUN_NAME = "bird_drone_yolov8_light"

YOLO_RUN_DIR = PROJECT_ROOT / "runs" / "detect" / YOLO_RUN_NAME
YOLO_WEIGHTS_BEST = YOLO_RUN_DIR / "weights" / "best.pt"
YOLO_WEIGHTS_LAST = YOLO_RUN_DIR / "weights" / "last.pt"

# Classifier location - change if your model file is elsewhere
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "bird_drone_mobilenetv2_tl.h5"

IMG_SIZE = (224, 224)  # classifier input size

# ---- Helper to select existing YOLO weight file ----
def find_yolo_weights():
    if YOLO_WEIGHTS_BEST.exists():
        return YOLO_WEIGHTS_BEST
    if YOLO_WEIGHTS_LAST.exists():
        return YOLO_WEIGHTS_LAST
    # fallback: search most recent run folder
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    if runs_dir.exists():
        runs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        for r in runs:
            cand_best = r / "weights" / "best.pt"
            cand_last = r / "weights" / "last.pt"
            if cand_best.exists():
                return cand_best
            if cand_last.exists():
                return cand_last
    return None

# ---- Load models ----
YOLO_WEIGHTS = find_yolo_weights()
if YOLO_WEIGHTS is None:
    raise FileNotFoundError("YOLO weights not found. Please run YOLO training or set YOLO_RUN_NAME correctly. "
                            f"Expected e.g. {YOLO_WEIGHTS_BEST} or {YOLO_WEIGHTS_LAST}")

if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(f"Classifier model not found at {CLASSIFIER_PATH}. Save your Keras model there or update CLASSIFIER_PATH.")

print("Loading YOLO model from:", YOLO_WEIGHTS)
yolo_model = YOLO(str(YOLO_WEIGHTS))

print("Loading classifier from:", CLASSIFIER_PATH)
classifier = load_model(str(CLASSIFIER_PATH))

CLASS_IDX_TO_NAME = {0: "bird", 1: "drone"}

# ---- Classification helper ----
def classify_crop(pil_crop):
    img = pil_crop.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(classifier.predict(arr)[0,0])
    class_idx = 1 if prob >= 0.5 else 0
    return prob, class_idx, CLASS_IDX_TO_NAME[class_idx]

# ---- Main pipeline ----
def run_pipeline(
    image_path_or_pil,
    yolo_conf_thresh=0.25,
    clf_conf_override=0.6,
    save_crops=False,
    imgsz=416
):
    """
    Runs YOLO detection + classifier with thresholds and crop saving.

    Args:
        image_path_or_pil: str | Path | PIL.Image
        yolo_conf_thresh: float -> ignore YOLO detections below this confidence
        clf_conf_override: float -> if classifier predicts drone >= this prob, override YOLO label
        save_crops: bool -> save detected crops to disk
        imgsz: int -> YOLO inference size (416 = faster on CPU)

    Returns:
        annotated_rgb: np.ndarray (RGB image)
        detections: list of dicts
    """

    # Load image
    if isinstance(image_path_or_pil, (str, Path)):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    img_np = np.array(img)

    # YOLO inference
    results = yolo_model.predict(source=img_np, imgsz=imgsz, verbose=False)
    detections = []

    crop_out_dir = PROJECT_ROOT / "runs" / "pipeline_crops"
    if save_crops:
        crop_out_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < yolo_conf_thresh:
                continue

            xyxy = box.xyxy[0].tolist()
            yolo_cls = int(box.cls[0])

            x1, y1, x2, y2 = map(int, xyxy)
            h, w, _ = img_np.shape

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))

            # classify crop
            prob, clf_idx, clf_name = classify_crop(crop)  # prob = P(drone)

            # Decide final label
            if prob >= clf_conf_override:
                final_label = "drone"
                final_conf = prob
            else:
                final_label = "bird" if clf_idx == 0 else "drone"
                final_conf = prob

            # Save crop
            if save_crops:
                crop.save(crop_out_dir / f"crop_{i}_{final_label}.jpg")

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "yolo_class_idx": yolo_cls,
                "yolo_conf": conf,
                "clf_class_idx": clf_idx,
                "clf_class_name": clf_name,
                "clf_prob": prob,
                "final_label": final_label,
                "final_conf": final_conf
            })

    # Annotate output image
    annotated = img_np.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['final_label']} (clf:{det['clf_prob']:.2f}, yolo:{det['yolo_conf']:.2f})"

        color = (0, 255, 0) if det['final_label'] == "drone" else (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections


# Quick CLI test
if __name__ == "__main__":
    test_img = next((PROJECT_ROOT / "data" / "Detection" / "Test" / "test" / "images").glob("*"), None)
    if test_img:
        ann, dets = run_pipeline(test_img)
        print("Detections:", dets)
        # save demo
        out_path = PROJECT_ROOT / "runs" / "pipeline_demo.jpg"
        Image.fromarray(ann).save(out_path)
        print("Annotated demo saved to:", out_path)
    else:
        print("No test image found at data/Detection/Test/test/images.")
