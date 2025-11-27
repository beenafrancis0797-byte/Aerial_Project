import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

DATA_YAML = "C:/Users/beena/Aerial_Project/data/Detection/data.yaml"

print("Using data.yaml:", DATA_YAML)

model = YOLO("yolov8n.pt")

results = model.train(
    data=DATA_YAML,
    imgsz=640,
    epochs=1,       # test only
    batch=4,
    device="cpu",
    name="bird_drone_test_run"
)

print("\nTest training complete!")
print("Check runs/detect/bird_drone_test_run")
