import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

DATA_YAML = "C:/Users/beena/Aerial_Project/data/Detection/data.yaml"

print("Using data.yaml:", DATA_YAML)

model = YOLO("yolov8n.pt")

results = model.train(
    data=DATA_YAML,
    imgsz=416,      # ↓ smaller than 640
    epochs=10,      # ↓ fewer epochs for now
    batch=4,        # small batch
    workers=0,      # avoids extra dataloader overhead on CPU
    device="cpu",
    name="bird_drone_yolov8_light"
)

print("\nTraining complete!")
print("Check runs/detect/bird_drone_yolov8_light")
