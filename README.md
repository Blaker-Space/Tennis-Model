# Pickleball Shot Recognition Using Pose Estimation and Transfer Learning

This repository contains our final project for the Fall 2025 Deep Learning course. Our goal is to classify pickleball shots (forehand, backhand, serve, neutral) using human pose estimation extracted from gameplay videos. The project uses MoveNet for pose extraction and a neural network classifier (Dense NN + RNN) for shot recognition.

---

## Project Overview

### 1. Annotation

Each video is manually annotated to mark frames where a shot occurs.

Example annotation CSV:

```
shot,frame
forehand,120
backhand,185
serve,310
neutral,402
```

### 2. Pose Extraction

We use MoveNet Lightning (TensorFlow Lite) to extract 17 keypoints per frame.

Command:

```
python extract_pose_as_features.py <video.mp4> <annotation.csv> <output_directory> --show
```

### 3. Shot Feature Construction

For each annotated shot, we extract ~1 second (30 frames) of pose coordinates and save them as CSV files.  
These CSVs form our dataset.

### 4. Shot Visualization

To verify pose sequences visually:

```
python visualize_features.py pickleball_shots/forehand_001.csv
```

This animates a stick-figure rendering of the pose.

### 5. Model Training

Training notebooks live in `model_training/`.

Models used:

- Dense Neural Network (baseline)
- GRU-based Recurrent Neural Network (sequence model)

The RNN provides significantly smoother and more accurate classification.

### 6. Evaluation

The notebook includes:

- Accuracy/loss curves
- Confusion matrix
- Interpretation of failure cases

---

## Repository Structure

```
├── annotations/              # Annotation CSV files
├── videos/                   # Raw videos (ignored by Git)
├── pickleball_shots/         # Pose-sequence CSVs used for training
├── extract_human_pose.py
├── extract_pose_as_features.py
├── visualize_features.py
├── model_training/           # Jupyter training notebooks
├── movenet.tflite            # Downloaded MoveNet model
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Create environment

```
python3 -m venv dl_env
source dl_env/bin/activate
```

### Install dependencies

```
pip install -r requirements.txt
```

### Download MoveNet

```
wget -q -O movenet.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite
```

---

## Running the Pipeline

### Extract shot features

```
python extract_pose_as_features.py <video.mp4> <annotation.csv> <output_dir> --show
```

### Visualize a shot sequence

```
python visualize_features.py <shot.csv>
```

### Train models

Open and run the notebooks inside `model_training/`.

---

## How This Meets the Final Project Requirements

- **Notebook Requirements**

  - Contains: Data → Model → Training → Evaluation
  - Extensive code comments
  - Required visualizations:
    1. Pipeline diagram
    2. Training curves (accuracy/loss)
    3. Performance interpretation + confusion matrix

- **Presentation Requirements**

  - Motivation and dataset overview
  - Architecture + pretrained backbone
  - Training setup and metrics
  - Results and visualizations
  - Limitations and future work

- **Code Repository**
  - Contains all scripts, model files, and notebooks used in the project

---

## Limitations

- Small dataset → limited generalization
- Pose extraction struggles with occlusion and unusual camera angles
- Only 4 shot categories supported

---

## Future Work

- Add more labeled data
- Use a larger backbone or multi-pose model
- Train for temporal localization
- Deploy real-time inference on mobile or webcam

---
