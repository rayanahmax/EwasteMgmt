"""
src/utils.py
------------
Shared utilities for the E-Waste Detection System.

Contains:
- CLASS_NAMES      : list of 7 e-waste category names
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
    "smartphone",   # 0
    "laptop",       # 1
    "battery",      # 2
    "pcb",          # 3
    "cables",       # 4
    "monitor",      # 5
    "ewaste_pile",  # 6
]

# Distinct, high-contrast BGR colors per class
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (  0, 200, 255),   # smartphone  – amber
    1: ( 60, 255,  60),   # laptop      – lime
    2: (  0,  80, 255),   # battery     – red-orange
    3: (255, 100,  20),   # pcb         – blue
    4: (180,   0, 255),   # cables      – purple
    5: ( 20, 220, 220),   # monitor     – teal
    6: (100, 100, 100),   # ewaste_pile – gray
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
    line_thickness: int = 2,
    font_scale: float = 0.55,
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
        label = f"{label_name}  {conf:.0%}"

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
        )
        label_y = max(y1 - 6, th + baseline)
        cv2.rectangle(
            annotated,
            (x1, label_y - th - baseline),
            (x1 + tw + 4, label_y + baseline),
            color,
            cv2.FILLED,
        )

        # Text
        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            line_thickness,
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
