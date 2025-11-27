# test_pipeline.py
from pathlib import Path
from pipeline_detection_classification import run_pipeline, PROJECT_ROOT

# find one test image from Detection test folder
test_folder = PROJECT_ROOT / "data" / "Detection" / "Test" / "test" / "images"
test_img = next(test_folder.glob("*"), None)

if test_img is None:
    print("No test image found at:", test_folder)
else:
    print("Using test image:", test_img)
    annotated_img, detections = run_pipeline(test_img)
    print("Detections:", detections)
    out_path = PROJECT_ROOT / "runs" / "pipeline_demo.jpg"
    from PIL import Image
    Image.fromarray(annotated_img).save(out_path)
    print("Annotated demo saved to:", out_path)
