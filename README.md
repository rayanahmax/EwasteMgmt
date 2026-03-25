# 🔋 E-Waste Detection System

A computer vision system that detects and classifies **electronic waste (e-waste)** items from images using **YOLOv8** object detection. Designed for environmental monitoring and recycling assistance.

---

## 🗂️ Project Structure

```
evs/
├── dataset/
│   ├── images/
│   │   ├── train/     ← Training images
│   │   └── val/       ← Validation images
│   └── labels/
│       ├── train/     ← YOLO .txt annotations (train)
│       └── val/       ← YOLO .txt annotations (val)
│
├── models/            ← Saved model weights (best.pt, last.pt)
│
├── src/
│   ├── train.py       ← Training pipeline
│   ├── detect.py      ← Inference + bounding box visualization
│   └── utils.py       ← Helper functions and class config
│
├── app/
│   └── streamlit_app.py  ← Interactive Streamlit demo
│
├── notebooks/
│   └── dataset_exploration.ipynb  ← EDA notebook
│
├── requirements.txt
├── data.yaml
└── README.md
```

---

## 🏷️ Target Classes

| ID | Class Name    | Description                          |
|----|---------------|--------------------------------------|
| 0  | smartphone    | Mobile phones, broken or whole       |
| 1  | laptop        | Notebooks, laptops, ultrabooks       |
| 2  | battery       | Li-ion, AA, car batteries            |
| 3  | pcb           | Printed circuit boards               |
| 4  | cables        | USB, power, HDMI, charger cables     |
| 5  | monitor       | Screens, displays, CRTs              |
| 6  | ewaste_pile   | Mixed/unsorted e-waste heaps         |

---

## ⚙️ Setup & Installation

### 1. Clone / open the project

```bash
cd C:\Users\LOQ\Documents\evs
```

### 2. Create a virtual environment

```bash
py -3 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The project requires Python 3.8+. On Windows, use `py -3` if `python` points to Python 2.

---

## 📦 Dataset Format

Labels use YOLO format — one `.txt` file per image:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized between `0.0` and `1.0`.

**Example:** A smartphone occupying the center:
```
0 0.5 0.5 0.3 0.4
```

Place images in `dataset/images/train/` and corresponding labels in `dataset/labels/train/`.

> **Tip:** Use [Roboflow](https://roboflow.com) to annotate, augment, and export datasets in YOLO format.

---

## 🚀 Training

```bash
py -3 src/train.py --data data.yaml --epochs 50 --imgsz 640
```

**Options:**

| Argument    | Default          | Description                        |
|-------------|------------------|------------------------------------|
| `--data`    | `data.yaml`      | Dataset config file                |
| `--weights` | `yolov8n.pt`     | Pretrained weights to start from   |
| `--epochs`  | `50`             | Number of training epochs          |
| `--imgsz`   | `640`            | Input image size                   |
| `--batch`   | `-1` (auto)      | Batch size                         |
| `--project` | `models/`        | Output directory for results       |

Trained weights are saved to `models/train/weights/best.pt`.

---

## 🔍 Detection (CLI)

Run inference on a single image or folder:

```bash
py -3 src/detect.py --source path/to/image.jpg
```

**Options:**

| Argument    | Default          | Description                          |
|-------------|------------------|--------------------------------------|
| `--source`  | *(required)*     | Image path, folder, or `0` for cam   |
| `--weights` | `models/best.pt` | Trained model weights                |
| `--conf`    | `0.25`           | Confidence threshold                 |
| `--save`    | flag             | Save annotated image to `output/`    |

---

## 🖥️ Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

- Upload any image
- Adjust confidence threshold
- View annotated detections and download results

---

## 📓 Dataset Exploration Notebook

```bash
jupyter notebook notebooks/dataset_exploration.ipynb
```

Explore class distribution, bounding box statistics, and augmentation previews.

---

## 🔮 Future Extensions

- ⚠️ **Hazard detection** – Swollen batteries, cracked screens
- ♻️ **Recycling value estimation** – Price estimates per detected item
- 🏷️ **Risk classification** – Toxicity scoring
- 📹 **Real-time detection** – Webcam support via OpenCV

---

## 📄 License

MIT License — for educational and research use.
