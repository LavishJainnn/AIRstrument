# AIRstrument

AIRstrument is a real-time computer vision and machine learning engine that transforms your webcam into a polyphonic digital instrument. By tracking your hand gestures in 3D space, the system classifies specific chord shapes and triggers zero-latency MIDI audio through your operating system's native synthesizer.

No physical instrument or specialized hardware required, just your hands and a webcam.

## How It Works Under the Hood

This isn't a simple color-tracker or bounded-box script. AIRstrument uses a highly optimized Machine Learning pipeline:

1. **Vision Tracking (MediaPipe):** Captures 42 spatial landmarks (21 per hand) asynchronously at 30 FPS.
2. **Mathematical Normalization:** To ensure the AI doesn't memorize hand size or screen position, all coordinates are mathematically shifted to the wrist (Translation Invariance) and scaled by the maximum fingertip distance (Scale Invariance). The result is a pure geometric vector in $\mathbb{R}^{84}$ space.
3. **Neural Inference (TensorFlow Lite):** The 84-dimensional vector is passed through a custom Multi-Layer Perceptron (MLP). The network expands the data to 128 dimensions, compresses it to 64, and outputs an 8-class probability distribution via Softmax to determine the current chord.
4. **State Machine & MIDI:** A frame-buffer debounce threshold prevents "flickering" during hand transitions. Once a chord shape is held for 5 consecutive frames, the engine fires digital `note_on` arrays to the OS synthesizer via the `mido` protocol.

## 🛠 Tech Stack

* **Computer Vision:** OpenCV, MediaPipe
* **Deep Learning:** TensorFlow (Keras), Scikit-Learn
* **Edge Inference:** TensorFlow Lite (`.tflite`)
* **Audio Engineering:** Mido, Python-RtMidi
* **Data Manipulation:** NumPy, Pandas

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/AIRstrument.git](https://github.com/yourusername/AIRstrument.git)
cd AIRstrument

```

**2. Set up a virtual environment (Recommended)**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# Currently not available on macOS/Linux/Android.

```

**3. Install dependencies**

```bash
pip install -r res.txt

```

**4. Download the required AI Models**

* Note: The script is designed to automatically download the Google `hand_landmarker.task` file on first run. Ensure you have an active internet connection.

## Usage

Run the main inference engine:

```bash
python air_piano.py

```

* Ensure your webcam is unobstructed and well-lit.
* Press `q` while the video window is active to safely terminate the script and silence the MIDI ports.

### Gesture Mapping (The Chords)

The neural network was trained on a sequential finger-counting taxonomy. Hold these shapes to trigger their respective major triads:

* **Idle / Silence:** Both hands closed in fists.
* **A Major:** Left hand: Index finger open (1).
* **B Major:** Left hand: Index and Middle fingers open (2).
* **C Major:** Left hand: Three fingers open (3).
* **D Major:** Left hand: Four fingers open (4).
* **E Major:** Left hand: All five fingers open (5).
* **F Major:** Left hand: Five fingers open + Right hand: Thumb open (6).
* **G Major:** Left hand: Five fingers open + Right hand: Thumb and Index open (7).

## Audio Customization (Instruments)

By default, the script routes MIDI arrays to the `Microsoft GS Wavetable Synth`. Currently a gaint piano only.

## 🤝 Contributing

Pull requests are welcome. For major architectural changes (like swapping the MLP for a sequential RNN/LSTM model to handle temporal transitions), please open an issue first to discuss the engineering approach.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)

```

Are you planning to push this repository publicly today, or do you need me to help draft the initial commit commands?

```
