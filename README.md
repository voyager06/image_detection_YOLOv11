# Document Element Detection with YOLOv11

This repository contains a Jupyter notebook implementation for detecting and creating bounding boxes around different elements in documents (figures, tables, and text) using YOLOv11n on the TFT-ID dataset.

![Document Detection Example](screenshot_example.png)

## 📋 Overview

This project demonstrates how to:
1. Set up and use YOLOv11n for document element detection
2. Fine-tune YOLOv11 on the TFT-ID (Text, Figure, Table ID) custom dataset
3. Evaluate model performance
4. Perform inference on new document images

YOLOv11 builds on advancements from previous YOLO versions with improved architectural designs, enhanced feature extraction, and optimized training methods. It achieves higher mAP scores with fewer parameters than previous versions.

## 🔧 Setup Requirements

- Python 3.x
- GPU support (recommended)
- Required packages:
  - ultralytics
  - supervision
  - roboflow
  - opencv-python
  - Pillow
  - IPython

## 🚀 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/document-element-detection.git
cd document-element-detection

# Install required packages
pip install ultralytics supervision roboflow opencv-python pillow
```

## 📊 Dataset

This project uses the TFT-ID dataset from Roboflow, which contains annotated document images with three classes:
- Figure
- Table
- Text

To access the dataset:
1. Create a Roboflow account
2. Set up your API key
3. Download the TFT-ID dataset (version 1)

## 🏋️ Training

The model is trained using YOLOv11n as the base model. The training process includes:

```bash
# Starting with pre-trained YOLOv11n weights
yolo task=detect mode=train model=yolo11n.pt data=/path/to/dataset/data.yaml epochs=10 imgsz=640 plots=True
```

Training configurations:
- Base model: YOLOv11n
- Image size: 640px
- Training epochs: 10
- Validation included during training

## 📈 Model Performance

The training process generates:
- Confusion matrix
- Precision-recall curves
- Validation batch predictions
- Training/validation loss curves

View detailed performance metrics in the `runs/detect/train/` directory after training.

## 🔍 Inference

You can perform inference using the trained model in several ways:

### Using the YOLO CLI

```bash
# Run inference on test images
yolo task=detect mode=predict model=/path/to/best.pt conf=0.50 source=/path/to/images save=True
```

### Using Python

```python
import cv2
import supervision as sv
from ultralytics import YOLO

# Load the custom model
model = YOLO('path/to/best.pt')

# Load image
image = cv2.imread('your_image.png')

# Perform inference
results = model(image, conf=0.50)[0]

# Create Detections with labels
detections = sv.Detections(
    xyxy=results.boxes.xyxy.cpu().numpy(),
    confidence=results.boxes.conf.cpu().numpy(),
    class_id=results.boxes.cls.cpu().numpy().astype(int)
)

# Get class names from the model
class_names = model.names

# Annotate and display the image
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=[f"{class_names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
)

# Save annotated image
cv2.imwrite("annotated_image.jpg", annotated_image)
```

## 📁 Project Structure

```
├── notebook.ipynb               # Main notebook with all the code
├── runs/                        # Generated during training
│   └── detect/
│       ├── train/               # Training results
│       │   ├── weights/
│       │   │   ├── best.pt      # Best model weights
│       │   │   └── last.pt      # Last model weights
│       │   ├── confusion_matrix.png
│       │   ├── results.png
│       │   └── val_batch0_pred.jpg
│       └── predict/             # Inference results
├── datasets/                    # Downloaded dataset location
└── README.md                    # This file
```

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [Roboflow](https://roboflow.com/) for dataset management
- [Supervision](https://github.com/roboflow/supervision) for annotation utilities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.# image_detection_YOLOv11
