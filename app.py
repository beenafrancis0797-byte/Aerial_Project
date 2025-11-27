import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from io import BytesIO

from pipeline_detection_classification import run_pipeline, PROJECT_ROOT

#------------------------------------------
# Streamlit Page Config
#------------------------------------------
st.set_page_config(
    page_title="Bird vs Drone Detection",
    page_icon="🛰️",
    layout="wide"
)

#------------------------------------------
# Header
#------------------------------------------
st.markdown("""
# 🛰️ Bird vs Drone Detection System  
A combined **YOLOv8 + MobileNetV2** intelligent aerial surveillance tool.

Upload an aerial image and the system will:

### ✔ Detect objects (YOLOv8)  
### ✔ Classify each object as **Bird** or **Drone**  
### ✔ Display bounding boxes + confidence scores  
""")

st.markdown("---")

#------------------------------------------
# Sidebar - Settings Panel
#------------------------------------------
st.sidebar.header("⚙️ Detection Settings")

yolo_conf = st.sidebar.slider(
    "YOLO Confidence Threshold",
    min_value=0.1, max_value=1.0, value=0.25, step=0.05
)

clf_override = st.sidebar.slider(
    "Classifier Drone Override Threshold",
    min_value=0.1, max_value=1.0, value=0.6, step=0.05,
    help="If classifier predicts drone probability ≥ this value, override YOLO label"
)

save_crops = st.sidebar.checkbox("Save detection crops", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("App Version: **2.0 Advanced UI**")

#------------------------------------------
# File Upload
#------------------------------------------
uploaded = st.file_uploader("📤 Upload an aerial image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 Original Image")
        st.image(image, use_column_width=True)

    with st.spinner("🔍 Running detection + classification..."):
        annotated, detections = run_pipeline(
            image_path_or_pil=image,
            yolo_conf_thresh=yolo_conf,
            clf_conf_override=clf_override,
            save_crops=save_crops
        )

    with col2:
        st.subheader("🎯 Detection Results")
        st.image(annotated, caption="Annotated Output", use_column_width=True)

        if detections:
            st.success(f"✅ {len(detections)} Object(s) Detected")

            birds = sum(1 for d in detections if d["final_label"] == "bird")
            drones = sum(1 for d in detections if d["final_label"] == "drone")

            st.info(f"**Birds: {birds}   |   Drones: {drones}**")
        else:
            st.warning("⚠ No objects detected with current thresholds.")

    #------------------------------------------
    # Detection Table
    #------------------------------------------
    st.markdown("---")
    st.subheader("📄 Detailed Detection Table")

    if detections:
        df = pd.DataFrame([
            {
                "Bounding Box": det["bbox"],
                "Final Label": det["final_label"],
                "YOLO Class": det["yolo_class_idx"],
                "YOLO Conf": round(det["yolo_conf"], 3),
                "Classifier Label": det["clf_class_name"],
                "Classifier Prob (Drone)": round(det["clf_prob"], 3),
            }
            for det in detections
        ])

        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 Confidence Breakdown")

        for i, det in enumerate(detections, start=1):
            st.markdown(f"#### Object {i}: **{det['final_label']}**")
            st.progress(det["clf_prob"])
            st.caption(
                f"Drone Probability = {det['clf_prob']:.3f} | YOLO Conf = {det['yolo_conf']:.3f}"
            )

    #------------------------------------------
    # Download / Save Results
    #------------------------------------------
    st.markdown("---")
    st.subheader("📥 Download Results")

    # Convert annotated image to buffer
    annotated_pil = Image.fromarray(annotated)
    buffer = BytesIO()
    annotated_pil.save(buffer, format="PNG")
    buffer.seek(0)

    # Download button
    st.download_button(
        label="📷 Download Annotated Image",
        data=buffer,
        file_name="annotated_output.png",
        mime="image/png"
    )

    # Auto-save results to project folder
    save_dir = PROJECT_ROOT / "runs" / "streamlit_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_img_path = save_dir / f"annotated_{timestamp}.png"
    out_csv_path = save_dir / f"detections_{timestamp}.csv"

    annotated_pil.save(out_img_path)

    if detections:
        df.to_csv(out_csv_path, index=False)

    st.success(f"Results saved to: `{save_dir}`")

else:
    st.info("⬆ Upload an image to begin.")
