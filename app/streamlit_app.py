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
import pandas as pd
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Allow imports from the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import CLASS_NAMES, CLASS_COLORS, load_model, draw_detections
from src.impact_analysis import get_ai_analysis, get_bulk_impact_summary

# Cache AI results to stay under the strict 5 RPM rate limit
@st.cache_data(show_spinner=False)
def cached_ai_analysis(class_name):
    return get_ai_analysis(class_name)

@st.cache_data(show_spinner=False)
def cached_bulk_summary(unique_names):
    return get_bulk_impact_summary(unique_names)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Waste Impact Analysis",
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

    /* Hero Styling */
    .hero-container {
        text-align: center;
        padding: 3rem 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .hero-container h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00e5ff, #76ff03, #ffea00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-container p {
        color: #8b949e;
        font-size: 1.2rem;
    }

    /* Detection card */
    .det-card {
        background: rgba(33,38,45,0.85);
        border: 1px solid rgba(48,54,61,0.9);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .det-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .det-badge {
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #000;
    }
    .det-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e6edf3;
    }
    
    /* Impact text */
    .impact-desc {
        background: rgba(0, 229, 255, 0.08);
        border-left: 4px solid #00e5ff;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-size: 0.95rem;
        color: #c9d1d9;
        font-style: italic;
    }

    /* Benefits Boxes */
    .benefit-container {
        display: flex;
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .benefit-box {
        flex: 1;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .benefit-env { background: rgba(118, 255, 3, 0.05); border-left: 5px solid #76ff03; }
    .benefit-eco { background: rgba(255, 234, 0, 0.05); border-left: 5px solid #ffea00; }
    
    .benefit-title {
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Global Info Cards */
    .fact-card {
        background: rgba(33,38,45,0.6);
        border: 1px solid rgba(48,54,61,0.5);
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
    }
    .fact-icon { font-size: 2rem; margin-bottom: 1rem; }
    .fact-text { font-size: 0.95rem; color: #8b949e; line-height: 1.6; }

    hr { border-color: rgba(48,54,61,0.4); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar (Left Side)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
    st.markdown("## ⚙️ Control Center")
    st.markdown("---")

    conf_threshold = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("IOU", 0.10, 0.90, 0.45, 0.05)

    st.markdown("---")
    st.markdown("### 📋 Classes")
    for name in CLASS_NAMES:
        st.sidebar.markdown(f"- **{name.title()}**")

    st.markdown("---")
    st.markdown("### 🤖 Analysis Mode")
    use_ai = st.toggle("Enable AI Analysis", value=True)
    
    default_weights = str(ROOT / "runs" / "detect" / "models" / "train2" / "weights" / "best.pt")
    weights_path = st.text_input("Model Path", value=default_weights)

# ─────────────────────────────────────────────────────────────────────────────
# Main Application Layout
# ─────────────────────────────────────────────────────────────────────────────

# TOP center title
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3.5rem; font-weight: 800; background: linear-gradient(90deg, #00e5ff, #76ff03, #ffea00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            E-Waste Detection & Impact Analysis
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Below title: Detection and analysis
st.markdown("### Detection and analysis")

# Then upload image
uploaded_file = st.file_uploader(
    "📂 Upload e-waste image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded_file:
    pil_image = Image.open(uploaded_file)
    bgr_image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    model = load_model(weights_path)
    
    with st.spinner("Analyzing image..."):
        results = model(bgr_image, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        
        annotated_bgr = draw_detections(bgr_image, results, conf_threshold=conf_threshold)
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= conf_threshold:
                    detections.append({
                        "name": CLASS_NAMES[cls_id],
                        "conf": conf,
                        "id": cls_id
                    })

    # Show scanned image
    st.image(annotated_pil, width='stretch')

    # Summary Section
    st.markdown("---")
    st.markdown("### summary:")
    
    if detections:
        num_objects = len(detections)
        avg_conf = sum(d['conf'] for d in detections) / num_objects
        unique_classes = sorted(list({d['name'] for d in detections}))
        num_species = len(unique_classes)
        
        # Specific formatting
        st.markdown(f"**Objects-{num_objects}**")
        st.markdown(f"**class:{', '.join(unique_classes)}**")
        st.markdown(f"**Accuracy-{avg_conf:.0%}**")
        st.markdown(f"**Species-{num_species}**")
        
        st.markdown("---")
        # Graph (below this)
        st.markdown("#### Material Composition Graph")
        
        # Aggregate analysis for the graph (using first detection as primary or bulk)
        # For simplicity and clarity, we'll show composition for each detected item in an expander 
        # but the request asks for "the graph" below the summary.
        # I'll provide an aggregated bar chart if multiple items exist, or 
        # just the primary detections' composition.
        
        all_comp = {}
        for d in detections:
            info = cached_ai_analysis(d['name'])
            for mat, perc in info['composition'].items():
                all_comp[mat] = all_comp.get(mat, 0) + perc
        
        # Normalize aggregated composition
        total = sum(all_comp.values())
        if total > 0:
            for mat in all_comp:
                all_comp[mat] = (all_comp[mat] / total) * 100

        comp_df = pd.DataFrame(
            list(all_comp.items()), 
            columns=['Material', 'Composition (%)']
        ).set_index('Material')
        
        st.bar_chart(comp_df, x_label="Material", y_label="%", color="#00e5ff", height=300)
        
        # AI Insight if enabled
        if use_ai:
            unique_names = sorted(list({d['name'] for d in detections}))
            with st.spinner("Generating AI Impact Analysis..."):
                ai_summary = cached_bulk_summary(unique_names)
            
            st.markdown("---")
            st.markdown("### Environmental & Economic Impact")
            st.info(ai_summary)
    else:
        st.warning("No objects detected above the confidence threshold.")
else:
    st.info("💡 **Ready to scan.** Upload an image above to begin detection and analysis.")

# Add footer spacing
st.markdown("<br><br>", unsafe_allow_html=True)
