# AIRstrument

### Gesture-controlled musical instrument using computer vision, machine learning, and MIDI

AIRstrument transforms a standard webcam into a playable digital instrument. It uses real-time hand landmark tracking to interpret hand gestures, classifies chord shapes with a lightweight neural network, and converts those predictions into MIDI note events.

The project combines **computer vision, geometric feature engineering, neural-network inference, temporal state handling, and MIDI control** into a real-time interaction system.

> **Play an instrument that doesn't physically exist.**

---

## Overview

Traditional digital instruments require a physical interface such as a keyboard, MIDI controller, or guitar.

AIRstrument replaces that physical interface with hand movement:

```text
                    Webcam
                       │
                       ▼
              MediaPipe Hand Tracking
                       │
                       ▼
          Landmark Normalization
                       │
                       ▼
             42-Dimensional Vector
                       │
                       ▼
           TensorFlow Lite MLP
                       │
                       ▼
             Chord Classification
                       │
                       ▼
          Gesture / State Processing
                       │
                       ▼
                 MIDI Events
                       │
                       ▼
             Software Synthesizer
                       │
                       ▼
                    Audio
```

The system currently recognizes the major chords **A, B, C, D, E, F, and G**, together with an **Idle** class.

---

## Key Features

- Real-time hand tracking using MediaPipe
- Webcam-based interaction with no specialized input hardware
- Scale- and translation-normalized hand landmark features
- Custom MLP classifier trained specifically for chord gestures
- TensorFlow Lite inference for lightweight runtime execution
- Temporal prediction stabilization to reduce gesture flicker
- MIDI chord triggering through `mido`
- `python-rtmidi` backend for MIDI communication
- Two-hand interaction for chord selection and strumming
- Real-time OpenCV visualization
- Custom dataset collection and model-training pipeline

---

## System Architecture

AIRstrument is divided into four main stages.

### 1. Hand Tracking

MediaPipe Hand Landmarker detects up to two hands and provides **21 landmarks per hand**.

The application runs the detector in live-stream mode and processes webcam frames asynchronously.

For the chord-classification pipeline, the relevant hand is represented using:

```text
21 landmarks × 2 coordinates (x, y)
= 42 input features
```

### 2. Feature Engineering

Raw pixel coordinates are not directly fed into the classifier.

For each detected hand:

1. Landmark coordinates are converted into pixel coordinates.
2. The wrist landmark is used as the origin.
3. All other coordinates are translated relative to the wrist.
4. The resulting coordinates are normalized by the maximum landmark distance.
5. The normalized `(x, y)` coordinates are flattened into a 42-dimensional vector.

This provides:

- **Translation invariance** — hand position in the camera frame has less influence.
- **Scale normalization** — different distances from the camera have less influence.
- **Compact representation** — only normalized 2D geometry is passed to the classifier.

### 3. Neural Network Inference

The gesture classifier is a fully connected Multi-Layer Perceptron.

```text
Input
42 features
   │
   ▼
Dense
128 neurons
ReLU
   │
   ▼
Dropout
0.30
   │
   ▼
Dense
64 neurons
ReLU
   │
   ▼
Dropout
0.20
   │
   ▼
Dense
8 neurons
Softmax
   │
   ▼
Chord class
```

The trained model is exported to TensorFlow Lite:

```text
Chords_classifier.tflite
```

The eight output classes correspond to:

```text
A
B
C
D
E
F
G
Idle
```

### 4. Temporal State + MIDI

A single incorrect prediction should not immediately change the active chord.

AIRstrument therefore maintains the previous prediction and requires the prediction to remain stable for multiple frames before changing the current chord.

Once the chord is established, the right-hand movement is used to trigger the chord through MIDI.

The right-hand index and middle fingertip trajectories are used to detect upward movement. Horizontal hand position is also used to influence MIDI velocity.

---

## Gesture Interface

The current gesture vocabulary maps sequential finger configurations to major chords.

| Gesture | Chord |
|---|---|
| No recognized chord | Idle |
| 1 | A Major |
| 2 | B Major |
| 3 | C Major |
| 4 | D Major |
| 5 | E Major |
| 5 + right-hand thumb | F Major |
| 5 + right-hand thumb + index | G Major |

The exact hand configuration is learned from the dataset rather than being hard-coded as a traditional finger-counting algorithm.

---

## MIDI Architecture

AIRstrument does not synthesize audio itself.

Instead, it generates MIDI messages and sends them to the system's available MIDI output.

The project uses:

- `mido` for MIDI message handling
- `python-rtmidi` as the MIDI backend

A chord is represented as a collection of MIDI note numbers.

For example:

```text
A Major → A + C# + E
B Major → B + D# + F#
C Major → C + E + G
...
```

The application sends:

```text
note_on
```

when a chord is triggered and:

```text
note_off
```

when the active chord is released or replaced.

The default instrument program is configured in the Python application and depends on the MIDI synthesizer available on the host system.

---

## Repository Structure

```text
AIRstrument/
│
├── AIRstrument.py
├── AIR_Piano.py
├── collecting_data.py
├── imp_mods.py
├── mlp.py
│
├── Chords_classifier.tflite
├── chords2.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

### File Responsibilities

| File | Purpose |
|---|---|
| `AIRstrument.py` | Main real-time AIRstrument application |
| `AIR_Piano.py` | Alternative piano-oriented interaction script |
| `collecting_data.py` | Webcam-based gesture dataset collection |
| `imp_mods.py` | Hand tracking, landmark processing, normalization, and kinematics |
| `mlp.py` | MLP training and TensorFlow Lite model export |
| `Chords_classifier.tflite` | Trained TensorFlow Lite classifier |
| `chords2.csv` | Gesture training dataset |
| `requirements.txt` | Python dependencies |

---

# Installation

## Prerequisites

Recommended environment:

- Python 3.10+
- Webcam
- Working audio output
- MIDI-capable software synthesizer or MIDI output
- Windows is currently the primary target environment for the default MIDI configuration

## 1. Clone the repository

```bash
git clone https://github.com/LavishJainnn/AIRstrument.git
cd AIRstrument
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project depends on:

```text
opencv-python
numpy
mediapipe
pandas
tensorflow
scikit-learn
mido
python-rtmidi
```

---

# Running AIRstrument

Start the main application:

```bash
python AIRstrument.py
```

The application will:

1. Initialize the webcam.
2. Initialize MediaPipe Hand Landmarker.
3. Load `Chords_classifier.tflite`.
4. Detect hand landmarks.
5. Normalize the classification hand.
6. Predict the current chord.
7. Stabilize the prediction.
8. Detect right-hand strumming movement.
9. Send the corresponding MIDI notes.

Press:

```text
q
```

to exit.

---

# MediaPipe Model

The project requires Google's MediaPipe Hand Landmarker model:

```text
hand_landmarker.task
```

`imp_mods.py` checks whether the file exists locally.

If it is not present, the application downloads the model automatically from Google's MediaPipe model storage.

Therefore, the first execution requires an active internet connection.

The downloaded model is intentionally not required to be committed to the repository.

---

# Training Pipeline

AIRstrument includes the complete pipeline used to create the gesture classifier.

```text
Webcam
   ↓
Hand Landmark Detection
   ↓
Feature Normalization
   ↓
CSV Dataset
   ↓
Label Encoding
   ↓
Train / Validation Split
   ↓
MLP Training
   ↓
Best Model
   ↓
TensorFlow Lite Conversion
   ↓
Chords_classifier.tflite
```

## Collecting Data

Run:

```bash
python collecting_data.py
```

The script creates or appends to:

```text
chords2.csv
```

The dataset contains:

```text
Label
L_x0
L_y0
L_x1
L_y1
...
L_x20
L_y20
```

This produces:

```text
1 label + 42 landmark features
```

### Data Collection Controls

| Key | Action |
|---|---|
| `a`–`z` | Start recording a class using that label |
| `n` | Record the `Ideal` class |
| `s` | Start / stop recording |
| `q` | Quit |

The collector records up to **1024 samples per recording session**.

For reliable classification, training data should contain natural variation in:

- Hand position
- Hand scale
- Finger configuration
- Camera distance
- Lighting
- Small changes in hand orientation

---

# Training the Classifier

Run:

```bash
python mlp.py
```

The training script:

1. Loads `chords2.csv`.
2. Separates labels and features.
3. Encodes class labels.
4. Converts labels to one-hot vectors.
5. Creates an 80/20 stratified train-validation split.
6. Trains the MLP.
7. Uses early stopping.
8. Saves the best Keras checkpoint.
9. Converts the trained model to TensorFlow Lite.

### Training Configuration

| Parameter | Value |
|---|---:|
| Input features | 42 |
| Hidden layer 1 | 128 |
| Hidden layer 2 | 64 |
| Output classes | 8 |
| Activation | ReLU |
| Output activation | Softmax |
| Dropout 1 | 0.30 |
| Dropout 2 | 0.20 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Maximum epochs | 200 |
| Batch size | 32 |
| Validation split | 20% |
| Early stopping patience | 15 |

The exported model is written to:

```text
Chords_classifier.tflite
```

---

# Performance Considerations

The runtime pipeline is designed to keep inference lightweight:

- MediaPipe performs hand landmark detection.
- Only normalized landmark coordinates are passed to the classifier.
- The neural network is a small fully connected model.
- TensorFlow Lite is used for inference.
- Temporal stabilization prevents rapid state changes from noisy frame-level predictions.

The application also requests a webcam capture configuration of:

```text
640 × 480
60 FPS
```

Actual camera performance depends on the webcam, operating system, MediaPipe processing time, and available compute resources.

---

# Customization

## Add or Change Chords

Chord definitions are stored in the application's chord map.

A chord can be represented by its MIDI note numbers:

```python
chord_map = {
    0: [45, 49, 52],
    1: [47, 51, 54],
    2: [48, 52, 55],
}
```

You can modify these values to experiment with:

- Minor chords
- Seventh chords
- Suspended chords
- Different voicings
- Different octaves

If the number or meaning of classifier classes changes, the training dataset and model must also be updated.

## Change the MIDI Instrument

The application sends a MIDI program-change message when it starts.

Change the program number to select a different instrument supported by the active MIDI synthesizer.

For example:

```python
midi_out.send(
    mido.Message('program_change', program=25)
)
```

The actual sound depends on the MIDI synthesizer and General MIDI implementation available on the host system.

---

# Limitations

The current implementation has several intentional limitations:

- The classifier recognizes a fixed set of gesture classes.
- The model is trained on a custom dataset and may require additional samples for different users.
- Hand occlusion can reduce tracking reliability.
- Poor lighting can affect landmark detection.
- MIDI output depends on the host system's available MIDI devices/synthesizer.
- The current interaction model is based on discrete chord gestures rather than continuous musical expression.
- The project currently targets a webcam-based desktop workflow.

These limitations also define several natural directions for future development.

---

# Roadmap

Potential improvements include:

### Gesture Recognition

- Expand the chord vocabulary
- Add minor, seventh, suspended, and extended chords
- Add user-specific calibration
- Improve robustness to occlusion
- Introduce confidence-aware prediction

### Temporal Modeling

Replace frame-level classification with a temporal model such as:

```text
MLP
  ↓
LSTM / GRU
  ↓
Temporal Transformer
```

This could allow the system to learn movement patterns rather than relying primarily on individual hand configurations.

### Musical Control

- Multiple instrument presets
- Configurable MIDI outputs
- Dynamic velocity curves
- Tempo-aware interactions
- Effects control
- Scale and mode selection
- User-configurable gesture mappings

### Platform

- Cross-platform MIDI configuration
- Native desktop interface
- Mobile deployment
- Browser-based version
- Wireless controller integration

---

# Development Notes

AIRstrument is intentionally structured as a small research/prototyping system rather than a large production framework.

The separation between:

```text
Data Collection
      ↓
Model Training
      ↓
Model Export
      ↓
Real-Time Inference
```

makes it possible to independently experiment with the dataset, classifier architecture, and interaction layer.

For major model changes, keep the input representation consistent between:

```text
collecting_data.py
        ↓
mlp.py
        ↓
AIRstrument.py
```

A mismatch between the feature representation used during training and inference will result in invalid predictions.

---

# Contributing

Contributions and experiments are welcome.

For substantial changes, especially changes to:

- Feature representation
- Gesture taxonomy
- Model architecture
- Temporal inference
- MIDI interaction
- Hand-tracking pipeline

it is recommended to open an issue first and describe the proposed approach.

A useful contribution should preserve reproducibility and clearly document changes to the training pipeline.

---

# License

This project is distributed under the MIT License.

---

## Author

**Lavish Jain**

Computer Science / Data Science Engineering

GitHub: [@LavishJainnn](https://github.com/LavishJainnn)

---

<p align="center">
  <strong>AIRstrument</strong><br>
  Turning hand gestures into music.
</p>
