"""
src/train.py
------------
Training pipeline for the E-Waste Detection System.

Usage
-----
python src/train.py --data data.yaml --epochs 50 --imgsz 640

The script:
  1. Loads a YOLOv8 pretrained model (yolov8n.pt by default).
  2. Trains it on the custom e-waste dataset defined in data.yaml.
  3. Saves the best weights to  models/train/weights/best.pt
"""

import argparse
import sys
from pathlib import Path


# ── CLI argument parsing ───────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model for E-Waste Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to the dataset YAML config file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help=(
            "Pretrained weights to start from. "
            "Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (pixels, square).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size. Use -1 for auto-batch.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="models",
        help="Root directory where training results are saved.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Sub-directory name inside --project for this run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use: '' (auto), 'cpu', '0' (GPU 0), '0,1' (multi-GPU).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable Albumentations augmentation pipeline.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved checkpoint.",
    )
    return parser.parse_args()


# ── Validation helpers ─────────────────────────────────────────────────────────
def validate_environment(args: argparse.Namespace) -> None:
    """Sanity-check paths and config before starting training."""
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: '{data_path}'")
        print("        Make sure you run training from the project root directory.")
        sys.exit(1)

    # Check that at least the train image folder exists
    import yaml  # PyYAML — already in requirements

    with open(data_path) as f:
        cfg = yaml.safe_load(f)

    train_dir = Path(cfg.get("train", "dataset/images/train"))
    if not train_dir.exists() or not any(train_dir.iterdir()):
        print(f"[WARNING] Training image folder is empty or missing: '{train_dir}'")
        print(
            "          Please add images to dataset/images/train/ "
            "and matching labels to dataset/labels/train/ before training."
        )
        print(
            "          You can download a ready-made dataset from "
            "https://roboflow.com and export in YOLO format."
        )
        sys.exit(1)


# ── Main training routine ──────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed.")
        print("        Run:  pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("  E-Waste Detection — Training Pipeline")
    print("=" * 60)
    print(f"  Model       : {args.weights}")
    print(f"  Dataset     : {args.data}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Image size  : {args.imgsz}px")
    print(f"  Batch       : {'auto' if args.batch == -1 else args.batch}")
    print(f"  Device      : {'auto' if not args.device else args.device}")
    print(f"  Output      : {args.project}/{args.name}/")
    print("=" * 60)

    # Load model
    model = YOLO(args.weights)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device if args.device else None,
        patience=args.patience,
        augment=args.augment,
        resume=args.resume,
        # Recommended hyperparameters for small datasets
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # Report best weights location
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    if best_weights.exists():
        print(f"  ✓ Training complete!")
        print(f"  ✓ Best weights saved to: {best_weights}")
        print(f"\n  Run detection with:")
        print(f"    python src/detect.py --source <image_path> --weights {best_weights}")
    else:
        print("  Training finished. Check the project directory for weights.")
    print("=" * 60)

    return results


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    validate_environment(args)
    train(args)
