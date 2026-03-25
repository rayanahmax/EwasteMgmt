"""
src/detect.py
-------------
Inference script for the E-Waste Detection System.

Usage
-----
# Single image:
python src/detect.py --source path/to/image.jpg

# Folder of images:
python src/detect.py --source path/to/folder/ --save

# Webcam (index 0):
python src/detect.py --source 0

# Custom weights + threshold:
python src/detect.py --source img.jpg --weights models/train/weights/best.pt --conf 0.4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ── CLI argument parsing ───────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E-Waste Detection — Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path, folder of images, or '0' for webcam.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained YOLO .pt weights file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0 – 1.0).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: '' (auto), 'cpu', '0' (GPU 0).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated images to output/ directory.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show image window (useful in headless environments).",
    )
    return parser.parse_args()


# ── Single image inference ─────────────────────────────────────────────────────
def run_on_image(
    model,
    img_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    save: bool,
    display: bool,
) -> list[dict]:
    """
    Run detection on a single image file and optionally display / save results.

    Returns
    -------
    list[dict]
        List of detected objects:
        { class_id, class_name, confidence, bbox_xyxy }
    """
    from src.utils import CLASS_NAMES, draw_detections

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[WARN] Could not read image: {img_path}")
        return []

    t0 = time.perf_counter()
    results = model(
        image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Draw detections
    annotated = draw_detections(image, results, conf_threshold=conf)

    # Collect detection info
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            if conf_val >= conf:
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": (
                            CLASS_NAMES[cls_id]
                            if cls_id < len(CLASS_NAMES)
                            else str(cls_id)
                        ),
                        "confidence": conf_val,
                        "bbox_xyxy": box.xyxy[0].tolist(),
                    }
                )

    # Console output
    print(f"\n[detect] {img_path.name}  •  {elapsed_ms:.1f} ms  •  {len(detections)} detection(s)")
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
        print(
            f"         [{d['class_id']}] {d['class_name']:<15s}  "
            f"conf={d['confidence']:.2f}  box=({x1},{y1},{x2},{y2})"
        )

    # Display
    if display:
        win_name = f"E-Waste Detection — {img_path.name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, annotated)
        print("         Press any key to continue…")
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)

    # Save
    if save:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"         Saved → {out_path}")

    return detections


# ── Webcam / video stream ──────────────────────────────────────────────────────
def run_on_webcam(model, conf: float, iou: float, imgsz: int) -> None:
    """Run real-time detection on webcam feed. Press 'q' to quit."""
    from src.utils import draw_detections

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0).")
        sys.exit(1)

    print("[detect] Webcam stream started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        annotated = draw_detections(frame, results, conf_threshold=conf)

        cv2.imshow("E-Waste Detection — Live", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Load model
    try:
        from src.utils import load_model
    except ImportError:
        # Allow running directly from project root without src on sys.path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.utils import load_model

    model = load_model(args.weights)
    model.conf = args.conf
    model.iou  = args.iou

    source = args.source

    # ── Webcam ──────────────────────────────────────────────────────────
    if source == "0" or source.isdigit():
        run_on_webcam(model, args.conf, args.iou, args.imgsz)
        return

    source_path = Path(source)

    # ── Folder ──────────────────────────────────────────────────────────
    if source_path.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        images = [p for p in source_path.iterdir() if p.suffix.lower() in image_exts]
        if not images:
            print(f"[ERROR] No images found in folder: {source_path}")
            sys.exit(1)
        print(f"[detect] Processing {len(images)} image(s) from '{source_path}'…\n")
        for img_path in sorted(images):
            run_on_image(
                model, img_path,
                conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                save=args.save, display=not args.no_display,
            )
        return

    # ── Single image ─────────────────────────────────────────────────────
    if not source_path.exists():
        print(f"[ERROR] Source not found: '{source_path}'")
        sys.exit(1)

    run_on_image(
        model, source_path,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        save=args.save, display=not args.no_display,
    )


if __name__ == "__main__":
    main()
