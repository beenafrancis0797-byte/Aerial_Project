# 🛰️ Aerial Project — Drone vs Bird Detection & Classification

This project performs **drone vs bird classification** using both:

- **Deep Learning image classification (CNN, MobileNetV2)**
- **Object detection (YOLOv8)**
- **A combined pipeline for real-time detection + classification**
- **A Streamlit web application**

---

## 📁 Project Structure

Aerial_Project/
│
├── Aerial_Project.ipynb                    # Main Jupyter Notebook (full project)
│
├── app/                                    # Streamlit web app
│   └── app.py
│
├── pipeline/                               # Combined detection + classification pipeline
│   └── pipeline_detection_classification.py
│
├── train_yolo.py                           # YOLO training script
├── test_yolo.py                            # YOLO inference script
├── test_pipeline.py                        # Pipeline testing script
│
├── yolov8n.pt                              # Pretrained YOLO model (base weights)
│
├── models/                                 # Saved trained models
│   ├── custom_cnn.h5                       # Custom CNN model
│   ├── bird_drone_mobilenetv2_tl.h5        # Transfer Learning (MobileNetV2) model
│   └── (more models if added later)
│
├── data/                                   # Dataset folder
│   ├── Classification/
│   └── Detection/
│
├── runs/                                   # YOLO training output
│   └── detect/
│        └── bird_drone_yolov8_light/
│             └── weights/
│                 └── best.pt
│
├── .ipynb_checkpoints/                     # Notebook auto-saves (ignore)
├── __pycache__/                            # Python cache (ignore)
│
└── README.md                               # GitHub documentation
