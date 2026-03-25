"""
app/streamlit_app.py
--------------------
Interactive Streamlit demo for the E-Waste Detection System.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Allow imports from the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import CLASS_NAMES, CLASS_COLORS, load_model, draw_detections

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Waste Detector",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – dark premium theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%);
    }

    /* Header */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero-header h1 {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00e5ff, #76ff03, #ffea00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-header p {
        color: #8b949e;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* Detection card */
    .det-card {
        background: rgba(33,38,45,0.85);
        border: 1px solid rgba(48,54,61,0.9);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .det-badge {
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #000;
        white-space: nowrap;
    }
    .det-label {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e6edf3;
        flex: 1;
    }
    .det-conf {
        font-size: 0.82rem;
        color: #8b949e;
    }

    /* Stat box */
    .stat-grid {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-box {
        flex: 1;
        background: rgba(33,38,45,0.85);
        border: 1px solid rgba(48,54,61,0.9);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00e5ff;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #8b949e;
        margin-top: 0.2rem;
    }

    /* Info box */
    .info-box {
        background: rgba(0, 229, 255, 0.06);
        border-left: 3px solid #00e5ff;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 1rem;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(13,17,23,0.95) !important;
    }

    /* Buttons */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #00e5ff22, #76ff0322) !important;
        border: 1px solid #00e5ff66 !important;
        color: #00e5ff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #00e5ff44, #76ff0344) !important;
        border-color: #00e5ff !important;
    }

    hr { border-color: rgba(48,54,61,0.6); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def bgr_to_hex(bgr: tuple) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


@st.cache_resource(show_spinner="Loading model…")
def get_model(weights_path: str):
    return load_model(weights_path)


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def image_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def run_inference(model, bgr_img: np.ndarray, conf: float, iou: float):
    return model(bgr_img, conf=conf, iou=iou, verbose=False)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    default_weights = str(ROOT / "models" / "best.pt")
    weights_path = st.text_input(
        "Model Weights Path",
        value=default_weights,
        help="Path to your trained .pt file. Defaults to yolov8n.pt if not found.",
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Detections below this score will be ignored.",
    )

    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Controls overlap allowed between boxes during Non-Maximum Suppression.",
    )

    st.markdown("---")
    st.markdown("### 🏷️ Class Legend")
    for idx, name in enumerate(CLASS_NAMES):
        color_hex = bgr_to_hex(CLASS_COLORS[idx])
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color_hex};border-radius:3px;margin-right:6px;"></span>'
            f'**{name}**',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem;color:#484f58;">'
        "E-Waste Detection System v1.0<br>"
        "Powered by YOLOv8 · Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <h1>♻️ E-Waste Detection System</h1>
        <p>Upload an image to identify and classify electronic waste using YOLOv8</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, PNG, BMP, WebP",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="info-box">
            💡 <strong>How to use:</strong><br>
            1. Upload an image containing e-waste items.<br>
            2. Adjust confidence / IoU thresholds in the sidebar if needed.<br>
            3. Detections will appear instantly below the image.<br><br>
            <em>No model trained yet? The system falls back to YOLOv8n pretrained weights 
            for a quick demo — results will be for general objects, not e-waste classes, 
            until you train on your dataset.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
pil_image = Image.open(uploaded_file)
bgr_image = pil_to_cv2(pil_image)

model = get_model(weights_path)

with st.spinner("Running detection…"):
    t0 = time.perf_counter()
    results = run_inference(model, bgr_image, conf_threshold, iou_threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000

annotated_bgr = draw_detections(bgr_image, results, conf_threshold=conf_threshold)
annotated_pil = cv2_to_pil(annotated_bgr)

# Gather detections
detections = []
if results.boxes is not None:
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if conf >= conf_threshold:
            bbox   = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_id":   cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                "confidence": conf,
                "bbox":       bbox,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Stats bar
# ─────────────────────────────────────────────────────────────────────────────
avg_conf = (
    sum(d["confidence"] for d in detections) / len(detections)
    if detections else 0.0
)
unique_classes = len({d["class_id"] for d in detections})

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("🔍 Total Detections", len(detections))
with col_b:
    st.metric("🏷️ Unique Classes", unique_classes)
with col_c:
    st.metric("🎯 Avg Confidence", f"{avg_conf:.0%}")
with col_d:
    st.metric("⚡ Inference Time", f"{elapsed_ms:.0f} ms")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Image columns
# ─────────────────────────────────────────────────────────────────────────────
col_orig, col_det = st.columns(2)

with col_orig:
    st.markdown("**📷 Original Image**")
    st.image(pil_image, use_container_width=True)

with col_det:
    st.markdown("**🎯 Detection Results**")
    st.image(annotated_pil, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Download button
# ─────────────────────────────────────────────────────────────────────────────
st.download_button(
    label="⬇️  Download Annotated Image",
    data=image_to_bytes(annotated_pil),
    file_name=f"ewaste_detected_{uploaded_file.name.rsplit('.', 1)[0]}.png",
    mime="image/png",
)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Detection Details
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Detection Details")

if not detections:
    st.warning(
        "No objects detected above the confidence threshold. "
        "Try lowering the threshold in the sidebar."
    )
else:
    # Visual cards
    for d in sorted(detections, key=lambda x: x["confidence"], reverse=True):
        color_hex = bgr_to_hex(CLASS_COLORS.get(d["class_id"], (180, 180, 180)))
        x1, y1, x2, y2 = d["bbox"]
        st.markdown(
            f"""
            <div class="det-card">
                <div class="det-badge" style="background:{color_hex};">
                    {'ID ' + str(d['class_id'])}
                </div>
                <div class="det-label">{d['class_name'].replace('_', ' ').title()}</div>
                <div class="det-conf">
                    Conf: <strong>{d['confidence']:.0%}</strong> &nbsp;|&nbsp;
                    Box: ({x1}, {y1}) → ({x2}, {y2})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Data table (expandable)
    with st.expander("📊 Show as Table", expanded=False):
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "Class ID":   d["class_id"],
                    "Class Name": d["class_name"],
                    "Confidence": f"{d['confidence']:.4f}",
                    "x1": d["bbox"][0],
                    "y1": d["bbox"][1],
                    "x2": d["bbox"][2],
                    "y2": d["bbox"][3],
                    "Width":  d["bbox"][2] - d["bbox"][0],
                    "Height": d["bbox"][3] - d["bbox"][1],
                }
                for d in sorted(detections, key=lambda x: x["confidence"], reverse=True)
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
