"""
src/utils.py
------------
Shared utilities for the E-Waste Detection System.

Contains:
- CLASS_NAMES      : list of 6 e-waste category names
- CLASS_COLORS     : BGR color for each class (used by OpenCV)
- load_model()     : loads a YOLO model from disk
- draw_detections(): draws bounding boxes + labels on a frame
- xyxy_to_xywh()  : converts bounding box formats
"""

import cv2
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# Class definitions
# ──────────────────────────────────────────────

CLASS_NAMES: list[str] = [
    "PCB",          # 0
    "adapter",      # 1
    "cable",        # 2
    "laptop",       # 3
    "mouse",        # 4
    "smartphone",   # 5
]

# Distinct, high-contrast BGR colors per class
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 100,  20),   # pcb         – blue
    1: (  0, 165, 255),   # adapter     – orange
    2: (180,   0, 255),   # cables      – purple
    3: ( 60, 255,  60),   # laptop      – lime
    4: ( 20, 220, 220),   # mouse       – teal
    5: (  0, 200, 255),   # smartphone  – amber
}


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────

def load_model(weights_path: str = "models/best.pt"):
    """
    Load a YOLO model from *weights_path*.

    Falls back to 'yolov8n.pt' (auto-downloaded) if the path does not exist,
    which is useful for first-run testing without a trained model.

    Parameters
    ----------
    weights_path : str
        Path to the .pt model file.

    Returns
    -------
    ultralytics.YOLO
        Loaded YOLO model instance.
    """
    from ultralytics import YOLO  # lazy import so utils can be imported cheaply

    path = Path(weights_path)
    if not path.exists():
        print(
            f"[utils] Weights not found at '{weights_path}'. "
            "Loading pretrained 'yolov8n.pt' instead."
        )
        return YOLO("yolov8n.pt")
    return YOLO(str(path))


# ──────────────────────────────────────────────
# Drawing utilities
# ──────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    results,
    conf_threshold: float = 0.25,
    line_thickness: int = 3,      # Thicker box
    font_scale: float = 0.8,       # Larger text
) -> np.ndarray:
    """
    Draw bounding boxes and class labels on *image* for all detections
    above *conf_threshold*.

    Parameters
    ----------
    image : np.ndarray
        BGR image as a NumPy array (H x W x 3).
    results : ultralytics.engine.results.Results
        Single YOLO result object (e.g. ``model(img)[0]``).
    conf_threshold : float
        Minimum confidence to draw a box.
    line_thickness : int
        Bounding-box border thickness in pixels.
    font_scale : float
        OpenCV font scale for label text.

    Returns
    -------
    np.ndarray
        Annotated BGR image.
    """
    annotated = image.copy()
    text_thickness = max(1, line_thickness - 1)

    if results.boxes is None or len(results.boxes) == 0:
        return annotated

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        label_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        # Bounding box corners
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Build label string
        label = f"{label_name.upper()} {conf:.0%}"

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        label_y = max(y1, th + baseline + 5)
        
        # Fill background rectangle for text
        cv2.rectangle(
            annotated,
            (x1, label_y - th - baseline - 5),
            (x1 + tw + 10, label_y + baseline),
            color,
            cv2.FILLED,
        )

        # Text
        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

    return annotated


# ──────────────────────────────────────────────
# BBox format helpers
# ──────────────────────────────────────────────

def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple:
    """Convert xyxy → (x_center, y_center, width, height)."""
    w = x2 - x1
    h = y2 - y1
    return x1 + w / 2, y1 + h / 2, w, h


def xywh_to_xyxy(xc: float, yc: float, w: float, h: float) -> tuple:
    """Convert (x_center, y_center, width, height) → xyxy."""
    return xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2


def normalize_bbox(x1, y1, x2, y2, img_w: int, img_h: int) -> tuple:
    """Normalize pixel xyxy coords to [0, 1] range."""
    xc, yc, w, h = xyxy_to_xywh(x1, y1, x2, y2)
    return xc / img_w, yc / img_h, w / img_w, h / img_h
