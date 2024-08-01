# Dual Object Detection with YOLOv8

## Overview

This repository contains code and resources for implementing dual object detection on both video and image formats using YOLOv8. The project leverages advanced machine learning techniques, NVIDIA GPUs, and Python to achieve high-precision object detection.

## Features

- **Dual Object Detection**: Supports object detection in both videos and static images.
- **State-of-the-Art Model**: Utilizes YOLOv8 for efficient and accurate object detection.
- **Dataset Management**: Employs Roboflow for easy dataset handling and augmentation.
- **GPU Acceleration**: Uses NVIDIA GPUs to accelerate training and inference.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Image Detection](#image-detection)
   - [Video Detection](#video-detection)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Acknowledgements](#acknowledgements)
7. [License](#license)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Dual-Object-Detection-YOLOv8.git
    cd Dual-Object-Detection-YOLOv8
    ```

2. **Install dependencies**:
    ```bash
    pip install ultralytics==8.0.196
    pip install roboflow
    ```

3. **Verify NVIDIA GPU**:
    ```bash
    !nvidia-smi
    ```

## Usage

### Image Detection

1. **Run the detection script on an image**:
    ```python
    from ultralytics import YOLO
    from IPython.display import display, Image

    !yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='/path/to/image.jpg' save=True
    display(Image(filename='/path/to/output/image.jpg', height=600))
    ```

### Video Detection

1. **Run the detection script on a video**:
    ```python
    !yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='/path/to/video.mp4' save=True
    ```

## Training

1. **Download the dataset using Roboflow**:
    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="your_api_key")
    project = rf.workspace("your_workspace").project("your_project")
    version = project.version(1)
    dataset = version.download("yolov8")
    ```

2. **Train the model**:
    ```python
    !yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True
    ```

## Evaluation

1. **Evaluate the model**:
    ```python
    !yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
    ```

## Results

- **Confusion Matrix**:
    ![Confusion Matrix](runs/detect/train/confusion_matrix.png)

- **Validation Batch Prediction**:
    ![Validation Batch Prediction](runs/detect/train/val_batch0_pred.jpg)

## Acknowledgements

This project utilizes:
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)

Special thanks to the open-source community for their contributions and resources.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
