# EMG Winter Soldier Arm

A 3D-printed robotic hand controlled by EMG (electromyography) signals from your forearm. Flex your muscles, and the hand moves. The system runs real-time gesture classification entirely on-device using an ESP32-S3, with no laptop required during inference.

<!-- TODO: Add a photo of the arm here -->
<!-- ![EMG Winter Soldier Arm](assets/arm_photo.jpg) -->

## How It Works

Four EMG sensors on your forearm pick up electrical signals from muscle contractions. The ESP32 samples these signals at 1 kHz using DMA, extracts features from sliding windows, and classifies them into gestures. The predicted gesture drives five servo motors (one per finger) to mirror your hand movement in real time.

### Gestures

| Gesture | Description |
|---------|-------------|
| Rest | Relaxed, hand open (neutral) |
| Fist | All fingers closed |
| Open | All fingers extended |
| Hook 'Em | Index and pinky out, others closed (🤘) |
| Thumbs Up | Thumb extended, others closed |

## System Architecture

```
EMG Sensors (x4)
      │
      ▼
┌─────────────────────────────────────────┐
│  ESP32-S3                               │
│                                         │
│  ADC + DMA (1 kHz per channel)          │
│       │                                 │
│       ▼                                 │
│  IIR Bandpass Filter (20-450 Hz)        │
│       │                                 │
│       ▼                                 │
│  Feature Extraction (69 features)       │
│  RMS, MAV, WL, ZC, SSC, AR, FFT,        │
│  band powers, cross-channel correlation │
│       │                                 │
│       ▼                                 │
│  Multi-Model Voting                     │
│  ┌─────┐  ┌──────────┐  ┌─────┐         │
│  │ LDA │  │ Ensemble │  │ MLP │         │
│  └──┬──┘  └────┬─────┘  └──┬──┘         │
│     └──────────┼───────────┘            │
│                ▼                        │
│  EMA Smoothing + Majority Vote          │
│  + Debounce                             │
│       │                                 │
│       ▼                                 │
│  Servo Driver (5 fingers, 50 Hz PWM)    │
└─────────────────────────────────────────┘
      │
      ▼
  Robotic Hand
```

## Features

- **On-device inference**: All classification runs on the ESP32-S3. No laptop in the loop.
- **Multi-model voting**: Three classifiers (LDA, 3-specialist ensemble, int8 MLP via TFLite Micro) vote on each prediction for robustness.
- **69 EMG features**: Time-domain (RMS, MAV, waveform length, zero crossings, slope sign changes, Hjorth parameters, autoregressive coefficients), frequency-domain (mean/median frequency, peak frequency, spectral band powers via FFT), and cross-channel correlation.
- **Adaptive smoothing**: EMA smoothing, sliding-window majority vote, and transition debounce prevent jittery output.
- **Z-score calibration**: Per-user calibration stored in NVS flash, so the system adapts to different forearm placements and muscle strengths.
- **Full training pipeline**: Python GUI for data collection, signal visualization, model training, and live prediction. Train a new model and export C header weights in one workflow.
- **BLE control**: Connect via Bluetooth to start/stop streaming, trigger calibration, or switch between on-device and laptop-side prediction.

## Tech Stack

**Firmware (C/C++)**
- ESP-IDF on ESP32-S3 (PlatformIO)
- FreeRTOS for task scheduling
- DMA-based ADC sampling at 1 kHz
- esp-dsp library for FFT
- TensorFlow Lite Micro for MLP inference
- LEDC PWM for servo control
- NimBLE for Bluetooth Low Energy
- NVS flash for calibration persistence

**Training Pipeline (Python)**
- scikit-learn for LDA and ensemble training
- TensorFlow/TFLite for MLP quantization (int8)
- NumPy, SciPy for signal processing
- CustomTkinter GUI for data collection and visualization
- Automated C header export for model weights

## Project Structure

```
EMG_Arm/                        # ESP32 firmware (PlatformIO project)
├── src/
│   ├── app/main.c              # State machine, BLE commands, multi-model voting
│   ├── config/config.h         # Pin definitions, constants, gesture enums
│   ├── core/
│   │   ├── inference.c/h       # LDA classifier, 69-feature extraction, IIR filter
│   │   ├── inference_ensemble.c/h  # 3-specialist LDA ensemble (TD/FD/CC)
│   │   ├── inference_mlp.cc/h  # Int8 MLP via TFLite Micro
│   │   ├── calibration.c/h     # Z-score calibration with NVS storage
│   │   ├── gestures.c/h        # Gesture definitions and finger mappings
│   │   ├── bicep.c/h           # Bicep curl detection
│   │   ├── model_weights.h     # Exported LDA weights
│   │   └── model_weights_ensemble.h  # Exported ensemble weights
│   ├── drivers/
│   │   ├── emg_sensor.c/h      # ADC + DMA driver
│   │   └── hand.c/h            # Per-finger servo control
│   └── hal/
│       └── servo_hal.c/h       # Low-level PWM servo driver
├── platformio.ini
└── partitions.csv

# Python training and data collection
emg_gui.py                      # Full GUI: collect data, train models, live predict
learning_data_collection.py     # Data collection pipeline and feature extraction
learning_emg_filtering.py       # Signal filtering experiments
train_ensemble.py               # 3-specialist ensemble trainer, exports C headers
train_mlp_tflite.py             # MLP training and TFLite int8 quantization
live_predict.py                 # Laptop-side live prediction over serial
serial_stream.py                # Serial communication with ESP32
requirements.txt                # Python dependencies
```

## Getting Started

### Firmware

1. Install [PlatformIO](https://platformio.org/)
2. Open the `EMG_Arm/` folder as a PlatformIO project
3. Build and flash to an ESP32-S3:
   ```
   pio run -t upload
   ```
4. Monitor serial output:
   ```
   pio device monitor -b 921600
   ```

### Training Pipeline

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Launch the GUI:
   ```
   python emg_gui.py
   ```
3. Collect training data (guided gesture prompts with live EMG visualization)
4. Train models (LDA, ensemble, MLP) from the GUI
5. Export weights to C headers for on-device deployment

## How the Ensemble Works

The ensemble uses three specialist LDA classifiers, each trained on a different feature subset:

1. **Time-domain specialist**: RMS, MAV, waveform length, zero crossings, slope sign changes
2. **Frequency-domain specialist**: Mean/median frequency, peak frequency, band powers
3. **Cross-channel specialist**: Correlation coefficients between EMG channels

A meta-LDA combines the three specialists' predictions into a final classification. This is more robust than any single model because different gestures are more separable in different feature spaces.

## Acknowledgments

Built by [Surya Balaji](https://github.com/SuryaUT) and [Aadi Pulipaka](https://github.com/pulipakaa24).
