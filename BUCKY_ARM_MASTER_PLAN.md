# Bucky Arm — EMG Gesture Control: Master Implementation Reference
> Version: 2026-03-01 | Target: ESP32-S3 N32R16V (Xtensa LX7 @ 240 MHz, 512 KB SRAM, 16 MB OPI PSRAM)
> Supersedes: META_EMG_RESEARCH_NOTES.md + BUCKY_ARM_IMPROVEMENT_PLAN.md
> Source paper: doi:10.1038/s41586-025-09255-w (PDF: C:/VSCode/Marvel_Projects/s41586-025-09255-w.pdf)

---

## TABLE OF CONTENTS

- [PART 0 — SYSTEM ARCHITECTURE & RESPONSIBILITY ASSIGNMENT](#part-0--system-architecture--responsibility-assignment)
  - [0.1 Who Does What](#01-who-does-what)
  - [0.2 Operating Modes](#02-operating-modes)
  - [0.3 FSM Reference (EMG_MAIN mode)](#03-fsm-reference-emg_main-mode)
  - [0.4 EMG_STANDALONE Boot Sequence](#04-emg_standalone-boot-sequence)
  - [0.5 New Firmware Changes for Architecture](#05-new-firmware-changes-for-architecture)
  - [0.6 New Python Script: live_predict.py](#06-new-python-script-live_predictpy)
  - [0.7 Firmware Cleanup: system_mode_t Removal](#07-firmware-cleanup-system_mode_t-removal)
- [PART I — SYSTEM FOUNDATIONS](#part-i--system-foundations)
  - [1. Hardware Specification](#1-hardware-specification)
  - [2. Current System Snapshot](#2-current-system-snapshot)
  - [2.1 Confirmed Firmware Architecture](#21--confirmed-firmware-architecture-from-codebase-exploration)
  - [2.2 Bicep Channel Subsystem](#22--bicep-channel-subsystem-ch3--adc_channel_9--gpio-10)
  - [3. What Meta Built — Filtered for ESP32](#3-what-meta-built--filtered-for-esp32)
  - [4. Current Code State + Known Bugs](#4-current-code-state--known-bugs)
- [PART II — TARGET ARCHITECTURE](#part-ii--target-architecture)
  - [5. Full Recommended Multi-Model Stack](#5-full-recommended-multi-model-stack)
  - [6. Compute Budget for Full Stack](#6-compute-budget-for-full-stack)
  - [7. Why This Architecture Works for 3-Channel EMG](#7-why-this-architecture-works-for-3-channel-emg)
- [PART III — GESTURE EXTENSIBILITY](#part-iii--gesture-extensibility)
  - [8. What Changes When Adding or Removing a Gesture](#8-what-changes-when-adding-or-removing-a-gesture)
  - [9. Practical Limits of 3-Channel EMG](#9-practical-limits-of-3-channel-emg)
  - [10. Specific Gesture Considerations](#10-specific-gesture-considerations)
- [PART IV — CHANGE REFERENCE](#part-iv--change-reference)
  - [11. Change Classification Matrix](#11-change-classification-matrix)
- [PART V — FIRMWARE CHANGES](#part-v--firmware-changes)
  - [Change A — DMA-Driven ADC Sampling](#change-a--dma-driven-adc-sampling)
  - [Change B — IIR Biquad Bandpass Filter](#change-b--iir-biquad-bandpass-filter)
  - [Change C — Confidence Rejection](#change-c--confidence-rejection)
  - [Change D — On-Device NVS Calibration](#change-d--on-device-nvs-calibration)
  - [Change E — int8 MLP via TFLM](#change-e--int8-mlp-via-tflm)
  - [Change F — Ensemble Inference Pipeline](#change-f--ensemble-inference-pipeline)
- [PART VI — PYTHON/TRAINING CHANGES](#part-vi--pythontraining-changes)
  - [Change 0 — Forward Label Shift](#change-0--forward-label-shift)
  - [Change 1 — Expanded Feature Set](#change-1--expanded-feature-set)
  - [Change 2 — Electrode Repositioning](#change-2--electrode-repositioning)
  - [Change 3 — Data Augmentation](#change-3--data-augmentation)
  - [Change 4 — Reinhard Compression](#change-4--reinhard-compression)
  - [Change 5 — Classifier Benchmark](#change-5--classifier-benchmark)
  - [Change 6 — Simplified MPF Features](#change-6--simplified-mpf-features)
  - [Change 7 — Ensemble Training](#change-7--ensemble-training)
- [PART VII — FEATURE SELECTION FOR ESP32 PORTING](#part-vii--feature-selection-for-esp32-porting)
- [PART VIII — MEASUREMENT AND VALIDATION](#part-viii--measurement-and-validation)
- [PART IX — EXPORT WORKFLOW](#part-ix--export-workflow)
- [PART X — REFERENCES](#part-x--references)

---

# PART 0 — SYSTEM ARCHITECTURE & RESPONSIBILITY ASSIGNMENT

> This section is the authoritative reference for what runs where. All implementation
> decisions in later parts should be consistent with this partition.

## 0.1 Who Does What

| Responsibility | Laptop (Python) | ESP32 |
|----------------|-----------------|-------|
| EMG sensor reading | — | ✓ `emg_sensor_read()` always |
| Raw data streaming (for collection) | Receives CSV, saves to HDF5 | Streams CSV over UART |
| Model training | ✓ `learning_data_collection.py` | — |
| Model export | ✓ `export_to_header()` → `model_weights.h` | Compiled into firmware |
| On-device inference | — | ✓ `inference_predict()` |
| Laptop-side live inference | ✓ `live_predict.py` (new script) | Streams ADC + executes received cmd |
| Arm actuation | — (sends gesture string back to ESP32) | ✓ `gestures_execute()` |
| Autonomous operation (no laptop) | Not needed | ✓ `EMG_STANDALONE` mode |
| Bicep flex detection | — | ✓ `bicep_detect()` (new, Section 2.2) |
| NVS calibration | — | ✓ `calibration.c` (Change D) |

**Key rule**: The laptop is never required for real-time arm control in production.
The laptop's role is: collect data → train model → export → flash firmware → done.
After that, the ESP32 operates completely independently.

---

## 0.2 Operating Modes

Controlled by `#define MAIN_MODE` in `config/config.h`.
The enum currently reads `enum {EMG_MAIN, SERVO_CALIBRATOR, GESTURE_TESTER}`.
A new value `EMG_STANDALONE` must be added.

| `MAIN_MODE` | When to use | Laptop required? | Entry point |
|-------------|-------------|-----------------|-------------|
| `EMG_MAIN` | Development sessions, data collection, monitored operation | Yes — UART handshake to start any mode | `appConnector()` in `main.c` |
| `EMG_STANDALONE` | **Fully autonomous deployment** — no laptop | **No** — boots directly into predict+control | `run_standalone_loop()` (new function in `main.c`) |
| `SERVO_CALIBRATOR` | Hardware setup, testing servo range of motion | Yes (serial input) | Inline in `app_main()` |
| `GESTURE_TESTER` | Testing gesture→servo mapping via keyboard | Yes (serial input) | Inline in `app_main()` |

**How to switch mode**: change `#define MAIN_MODE` in `config.h` and reflash.

**To add `EMG_STANDALONE` to `config.h`** (1-line change):
```c
// config.h line 19 — current:
enum {EMG_MAIN, SERVO_CALIBRATOR, GESTURE_TESTER};

// Update to:
enum {EMG_MAIN, SERVO_CALIBRATOR, GESTURE_TESTER, EMG_STANDALONE};
```

---

## 0.3 FSM Reference (EMG_MAIN mode)

The `device_state_t` enum in `main.c` and the `command_t` enum control all transitions.
Currently: `{STATE_IDLE, STATE_CONNECTED, STATE_STREAMING, STATE_PREDICTING}`.
A new state `STATE_LAPTOP_PREDICT` must be added (see Section 0.5).

```
STATE_IDLE
  └─ {"cmd":"connect"} ──────────────────────────► STATE_CONNECTED
                                                         │
                               {"cmd":"start"} ──────────┤
                                                         │    STATE_STREAMING
                                                         │    ESP32 sends raw ADC CSV at 1kHz
                                                         │    Laptop: saves to HDF5 (data collection)
                                                         │    Laptop: trains model → exports model_weights.h
                                                         │    ◄──── {"cmd":"stop"} ────────────────────┘
                                                         │
                        {"cmd":"start_predict"} ─────────┤
                                                         │    STATE_PREDICTING
                                                         │    ESP32: inference_predict() on-device
                                                         │    ESP32: gestures_execute()
                                                         │    Laptop: optional UART monitor only
                                                         │    ◄──── {"cmd":"stop"} ────────────────────┘
                                                         │
                   {"cmd":"start_laptop_predict"} ───────┘
                                                              STATE_LAPTOP_PREDICT  [NEW]
                                                              ESP32: streams raw ADC CSV (same as STREAMING)
                                                              Laptop: runs live_predict.py inference
                                                              Laptop: sends {"gesture":"fist"} back
                                                              ESP32: executes received gesture command
                                                              ◄──── {"cmd":"stop"} ────────────────────┘

All active states:
  {"cmd":"stop"}       → STATE_CONNECTED
  {"cmd":"disconnect"} → STATE_IDLE
  {"cmd":"connect"}    → STATE_CONNECTED  (from any state — reconnect)
```

**Convenience table of commands and their effects:**

| JSON command | Valid from state | Result |
|---|---|---|
| `{"cmd":"connect"}` | Any | → `STATE_CONNECTED` |
| `{"cmd":"start"}` | `STATE_CONNECTED` | → `STATE_STREAMING` |
| `{"cmd":"start_predict"}` | `STATE_CONNECTED` | → `STATE_PREDICTING` |
| `{"cmd":"start_laptop_predict"}` | `STATE_CONNECTED` | → `STATE_LAPTOP_PREDICT` (new) |
| `{"cmd":"stop"}` | `STREAMING/PREDICTING/LAPTOP_PREDICT` | → `STATE_CONNECTED` |
| `{"cmd":"disconnect"}` | Any active state | → `STATE_IDLE` |

---

## 0.4 EMG_STANDALONE Boot Sequence

No UART handshake. No laptop required. Powers on → predicts → controls arm.

```
app_main() switch MAIN_MODE == EMG_STANDALONE:
  │
  ├── hand_init()            // servos
  ├── emg_sensor_init()      // ADC setup
  ├── inference_init()       // clear window buffer, reset smoothing state
  ├── calibration_init()     // load NVS z-score params (Change D)
  │       └── if not found in NVS:
  │               collect 120 REST windows (~3s at 25ms hop)
  │               call calibration_update() to compute and store stats
  ├── bicep_load_threshold() // load NVS bicep threshold (Section 2.2)
  │       └── if not found:
  │               collect 3s of still bicep data
  │               call bicep_calibrate() and bicep_save_threshold()
  │
  └── run_standalone_loop()  ← NEW function (added to main.c)
        while (1):
          emg_sensor_read(&sample)
          inference_add_sample(sample.channels)
          if stride_counter++ >= INFERENCE_HOP_SIZE:
            stride_counter = 0
            gesture_t g = inference_get_gesture_enum(inference_predict(&conf))
            gestures_execute(g)
            bicep_state_t b = bicep_detect()
            // (future: bicep_actuate(b))
          vTaskDelay(1)
```

`run_standalone_loop()` is structurally identical to `run_inference_loop()` in `EMG_MAIN`,
minus all UART state-change checking and telemetry prints. It runs forever until power-off.

**Where to add**: New function `run_standalone_loop()` in `app/main.c`, plus a new case
in the `app_main()` switch block:
```c
case EMG_STANDALONE:
    run_standalone_loop();
    break;
```

---

## 0.5 New Firmware Changes for Architecture

These changes are needed to implement the architecture above. They are **structural**
(not accuracy improvements) and should be done before any other changes.

### S1 — Add `EMG_STANDALONE` to `config.h`

**File**: `EMG_Arm/src/config/config.h`, line 19
```c
// Change:
enum {EMG_MAIN, SERVO_CALIBRATOR, GESTURE_TESTER};
// To:
enum {EMG_MAIN, SERVO_CALIBRATOR, GESTURE_TESTER, EMG_STANDALONE};
```

### S2 — Add `STATE_LAPTOP_PREDICT` to FSM (`main.c`)

**File**: `EMG_Arm/src/app/main.c`

```c
// In device_state_t enum — add new state:
typedef enum {
  STATE_IDLE = 0,
  STATE_CONNECTED,
  STATE_STREAMING,
  STATE_PREDICTING,
  STATE_LAPTOP_PREDICT,  // ← ADD: streams ADC to laptop, executes laptop's gesture commands
} device_state_t;

// In command_t enum — add new command:
typedef enum {
  CMD_NONE = 0,
  CMD_CONNECT,
  CMD_START,
  CMD_START_PREDICT,
  CMD_START_LAPTOP_PREDICT,  // ← ADD
  CMD_STOP,
  CMD_DISCONNECT,
} command_t;
```

**In `parse_command()`** — add detection (place BEFORE the `"start"` check to avoid prefix collision):
```c
} else if (strncmp(value_start, "start_laptop_predict", 20) == 0) {
    return CMD_START_LAPTOP_PREDICT;
} else if (strncmp(value_start, "start_predict", 13) == 0) {
    return CMD_START_PREDICT;
} else if (strncmp(value_start, "start", 5) == 0) {
    return CMD_START;
```

**In `serial_input_task()` FSM switch** — add to `STATE_CONNECTED` block:
```c
} else if (cmd == CMD_START_LAPTOP_PREDICT) {
    g_device_state = STATE_LAPTOP_PREDICT;
    printf("[STATE] CONNECTED -> LAPTOP_PREDICT\n");
    xQueueSend(g_cmd_queue, &cmd, 0);
}
```

**Add to the active-state check** in `serial_input_task()`:
```c
case STATE_STREAMING:
case STATE_PREDICTING:
case STATE_LAPTOP_PREDICT:  // ← ADD to the case list
    if (cmd == CMD_STOP) { ... }
```

**New function `run_laptop_predict_loop()`** (add alongside `stream_emg_data()` and `run_inference_loop()`):
```c
/**
 * @brief Laptop-mediated prediction loop (STATE_LAPTOP_PREDICT).
 *
 * Streams raw ADC CSV to laptop for inference.
 * Simultaneously reads gesture commands sent back by laptop.
 * Executes received gesture immediately.
 *
 * Laptop sends: {"gesture":"fist"}\n  OR  {"gesture":"rest"}\n  etc.
 * ESP32 parses the "gesture" field and calls inference_get_gesture_enum() + gestures_execute().
 */
static void run_laptop_predict_loop(void) {
    emg_sample_t sample;
    char cmd_buf[64];
    int cmd_idx = 0;

    printf("{\"status\":\"info\",\"msg\":\"Laptop-predict mode started\"}\n");

    while (g_device_state == STATE_LAPTOP_PREDICT) {
        // 1. Send raw ADC sample (same format as STATE_STREAMING)
        emg_sensor_read(&sample);
        printf("%u,%u,%u,%u\n", sample.channels[0], sample.channels[1],
               sample.channels[2], sample.channels[3]);

        // 2. Non-blocking read of any incoming gesture command from laptop
        //    (serial_input_task already handles FSM commands; this handles gesture commands)
        //    Note: getchar() is non-blocking when there is no data (returns EOF).
        //    Gesture messages from laptop look like: {"gesture":"fist"}\n
        int c = getchar();
        if (c != EOF && c != 0xFF) {
            if (c == '\n' || c == '\r') {
                if (cmd_idx > 0) {
                    cmd_buf[cmd_idx] = '\0';
                    // Parse {"gesture":"<name>"} — look for "gesture" field
                    const char *g = strstr(cmd_buf, "\"gesture\"");
                    if (g) {
                        const char *v = strchr(g, ':');
                        if (v) {
                            v++;
                            while (*v == ' ' || *v == '"') v++;
                            // Extract gesture name up to closing quote
                            char name[32] = {0};
                            int ni = 0;
                            while (*v && *v != '"' && ni < 31) name[ni++] = *v++;
                            name[ni] = '\0';
                            // Map name to enum and execute (reuse inference mapping)
                            gesture_t gesture = (gesture_t)inference_get_gesture_enum_by_name(name);
                            if (gesture != GESTURE_NONE) {
                                gestures_execute(gesture);
                            }
                        }
                    }
                    cmd_idx = 0;
                }
            } else if (cmd_idx < (int)sizeof(cmd_buf) - 1) {
                cmd_buf[cmd_idx++] = (char)c;
            } else {
                cmd_idx = 0;
            }
        }

        vTaskDelay(1);
    }
}
```

**Note**: `inference_get_gesture_enum_by_name(const char *name)` is just the existing
`inference_get_gesture_enum(int class_idx)` refactored to accept a string directly
(bypassing the class_idx lookup). Alternatively, keep the existing function and add a
simple wrapper — the string matching logic already exists in `inference.c`:
```c
// Simpler: reuse the existing strcmp chain in inference_get_gesture_enum()
// by passing the name through a helper that returns the gesture_t directly.
// Add to inference.c / inference.h:
gesture_t inference_get_gesture_by_name(const char *name);
// (same strcmp logic as inference_get_gesture_enum, but returns gesture_t directly)
```

**In `state_machine_loop()`** — add the new state:
```c
static void state_machine_loop(void) {
    command_t cmd;
    const TickType_t poll_interval = pdMS_TO_TICKS(50);
    while (1) {
        if      (g_device_state == STATE_STREAMING)        stream_emg_data();
        else if (g_device_state == STATE_PREDICTING)       run_inference_loop();
        else if (g_device_state == STATE_LAPTOP_PREDICT)   run_laptop_predict_loop();  // ← ADD
        xQueueReceive(g_cmd_queue, &cmd, poll_interval);
    }
}
```

**In `app_main()` switch** — add the standalone case:
```c
case EMG_STANDALONE:
    run_standalone_loop();  // new function — see Section 0.4
    break;
```

---

## 0.6 New Python Script: `live_predict.py`

**Location**: `C:/VSCode/Marvel_Projects/Bucky_Arm/live_predict.py` (new file)
**Purpose**: Laptop-side live inference. Reads raw ADC stream from ESP32, runs the Python
classifier, sends gesture commands back to ESP32 for arm control.
**When to use**: `EMG_MAIN` + `STATE_LAPTOP_PREDICT` — useful for debugging and comparing
laptop accuracy vs on-device accuracy before flashing a new model.

```python
"""
live_predict.py — Laptop-side live EMG inference for Bucky Arm.

Connects to ESP32, requests STATE_LAPTOP_PREDICT, reads raw ADC CSV,
runs the trained Python classifier, sends gesture commands back to ESP32.

Usage:
    python live_predict.py --port COM3 --model path/to/saved_model/
"""
import argparse
import time
import numpy as np
import serial
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from learning_data_collection import (
    EMGClassifier, EMGFeatureExtractor, SessionStorage, HAND_CHANNELS,
    WINDOW_SIZE_SAMPLES, HOP_SIZE_SAMPLES, NUM_CHANNELS,
)

BAUD_RATE    = 921600
CALIB_SEC    = 3.0          # seconds of REST to collect for normalization at startup
CALIB_LABEL  = "rest"       # label used during calibration window

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",  required=True, help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    p.add_argument("--model", required=True, help="Path to saved EMGClassifier model directory")
    return p.parse_args()

def handshake(ser):
    """Send connect command, wait for ack."""
    ser.write(b'{"cmd":"connect"}\n')
    deadline = time.time() + 5.0
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if "ack_connect" in line:
            print(f"[Handshake] Connected: {line}")
            return True
    raise RuntimeError("No ack_connect received within 5s")

def collect_calibration_windows(ser, n_windows, window_size, hop_size, n_channels):
    """Collect n_windows worth of REST data for normalization calibration."""
    print(f"[Calib] Collecting {n_windows} REST windows — hold arm still...")
    raw_buffer = np.zeros((window_size, n_channels), dtype=np.float32)
    windows = []
    sample_count = 0
    while len(windows) < n_windows:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        try:
            vals = [float(v) for v in line.split(",")]
            if len(vals) != n_channels:
                continue
        except ValueError:
            continue
        raw_buffer = np.roll(raw_buffer, -1, axis=0)
        raw_buffer[-1] = vals
        sample_count += 1
        if sample_count >= window_size and sample_count % hop_size == 0:
            windows.append(raw_buffer.copy())
    print(f"[Calib] Collected {len(windows)} windows. Computing normalization stats...")
    return np.array(windows)  # (n_windows, window_size, n_channels)

def main():
    args = parse_args()

    # Load trained classifier
    print(f"[Init] Loading classifier from {args.model}...")
    classifier = EMGClassifier()
    classifier.load(Path(args.model))
    extractor = classifier.feature_extractor

    ser = serial.Serial(args.port, BAUD_RATE, timeout=1.0)
    time.sleep(0.5)
    ser.reset_input_buffer()

    handshake(ser)

    # Request laptop-predict mode
    ser.write(b'{"cmd":"start_laptop_predict"}\n')
    print("[Control] Entered STATE_LAPTOP_PREDICT")

    # Calibration: collect 3s of REST for session normalization
    n_calib_windows = max(10, int(CALIB_SEC * 1000 / (HOP_SIZE_SAMPLES)))
    calib_raw = collect_calibration_windows(
        ser, n_calib_windows, WINDOW_SIZE_SAMPLES, HOP_SIZE_SAMPLES, NUM_CHANNELS
    )
    calib_features = extractor.extract_features_batch(calib_raw)
    calib_mean = calib_features.mean(axis=0)
    calib_std  = np.where(calib_features.std(axis=0) > 1e-6,
                          calib_features.std(axis=0), 1e-6)
    print("[Calib] Done. Starting live prediction...")

    # Live prediction loop
    raw_buffer   = np.zeros((WINDOW_SIZE_SAMPLES, NUM_CHANNELS), dtype=np.float32)
    sample_count = 0
    last_gesture = None

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            # Skip JSON telemetry lines from ESP32
            if line.startswith("{"):
                continue

            try:
                vals = [float(v) for v in line.split(",")]
                if len(vals) != NUM_CHANNELS:
                    continue
            except ValueError:
                continue

            # Slide window
            raw_buffer = np.roll(raw_buffer, -1, axis=0)
            raw_buffer[-1] = vals
            sample_count += 1

            if sample_count >= WINDOW_SIZE_SAMPLES and sample_count % HOP_SIZE_SAMPLES == 0:
                # Extract features and normalize with session stats
                feat = extractor.extract_features_window(raw_buffer)
                feat = (feat - calib_mean) / calib_std

                proba = classifier.model.predict_proba([feat])[0]
                class_idx = int(np.argmax(proba))
                gesture_name = classifier.label_names[class_idx]
                confidence   = float(proba[class_idx])

                # Send gesture command to ESP32
                cmd = f'{{"gesture":"{gesture_name}"}}\n'
                ser.write(cmd.encode("utf-8"))

                if gesture_name != last_gesture:
                    print(f"[Predict] {gesture_name:12s}  conf={confidence:.2f}")
                    last_gesture = gesture_name

    except KeyboardInterrupt:
        print("\n[Stop] Sending stop command...")
        ser.write(b'{"cmd":"stop"}\n')
        ser.close()

if __name__ == "__main__":
    main()
```

**Dependencies** (add to a `requirements.txt` in `Bucky_Arm/` if not already there):
```
pyserial
numpy
scikit-learn
```

---

## 0.7 Firmware Cleanup: `system_mode_t` Removal

`config.h` lines 94–100 define a `system_mode_t` typedef that is **not referenced anywhere**
in the firmware. It predates the current `device_state_t` FSM in `main.c` and conflicts
conceptually with it. Remove before starting implementation work.

**File**: `EMG_Arm/src/config/config.h`
**Remove** (lines 93–100):
```c
/**
 * @brief System operating modes.
 */
typedef enum {
    MODE_IDLE = 0,      /**< Waiting for commands */
    MODE_DATA_STREAM,   /**< Streaming EMG data to laptop */
    MODE_COMMAND,       /**< Executing gesture commands from laptop */
    MODE_DEMO,          /**< Running demo sequence */
    MODE_COUNT
} system_mode_t;
```
No other file references `system_mode_t` — the deletion is safe and requires no other changes.

---

# PART I — SYSTEM FOUNDATIONS

## 1. Hardware Specification

### ESP32-S3 N32R16V — Confirmed Hardware

| Resource | Spec | Implication |
|----------|------|-------------|
| CPU | Dual-core Xtensa LX7 @ 240 MHz | Pin inference to Core 1, sampling to Core 0 |
| SIMD | PIE 128-bit vector extension | esp-dsp exploits this for FFT, biquad, dot-product |
| Internal SRAM | ~512 KB | All hot-path buffers, model weights, inference state |
| OPI PSRAM | 16 MB (~80 MB/s) | ADC ring buffer, raw window storage — not hot path |
| Flash | 32 MB | Code + read-only model flatbuffers (TFLM path) |
| ADC | 2× SAR ADC, 12-bit, continuous DMA mode | Change A: use `adc_continuous` driver |

**Memory rules**:
- Tag inference code: `IRAM_ATTR` — prevents cache miss stalls
- Tag large ring buffers: `EXT_RAM_BSS_ATTR` — pushes to PSRAM automatically
- Never run hot-path loops from PSRAM (latency varies; ~10× slower than SRAM)

### Espressif Acceleration Libraries

| Library | Accelerates | Key Functions |
|---------|-------------|---------------|
| **esp-dsp** | IIR biquad, FFT (up to 4096-pt), vector dot-product, matrix ops — PIE SIMD | `dsps_biquad_f32`, `dsps_fft2r_fc32`, `dsps_dotprod_f32` |
| **esp-nn** | int8 FC, depthwise/pointwise Conv, activations — SIMD optimized | Used internally by esp-dl |
| **esp-dl** | High-level int8 inference: MLP, Conv1D, LSTM; activation buffer management | Small MLP / tiny CNN deployment |
| **TFLite Micro** | Standard int8 flatbuffer inference, tensor arena (static alloc) | Keras → TFLite → int8 workflow |

### Real-Time Budget (1000 Hz, 25ms hop)

| Stage | Cost | Notes |
|-------|------|-------|
| ADC DMA sampling | ~0 µs | Hardware; CPU-free |
| IIR biquad (3 ch, 2 stages) | <100 µs | `dsps_biquad_f32` |
| Feature extraction (69 feat) | ~1,200 µs | FFT-based features dominate |
| 3 specialist LDAs | ~150 µs | `dsps_dotprod_f32` per class |
| Meta-LDA (15 inputs) | ~10 µs | 75 MACs total |
| int8 MLP fallback [69→32→16→5] | ~250 µs | esp-nn FC kernels |
| Post-processing | <50 µs | EMA, vote, debounce |
| **Total (full ensemble)** | **~1,760 µs** | **14× margin within 25ms** |

### Hard No-Gos

| Technique | Why |
|-----------|-----|
| Full MPF with matrix logarithm | Eigendecomposition per window; fragile float32; no SIMD path |
| Conv1D(16→512) + 3×LSTM(512) | ~4 MB weights; LSTM sequential dependency — impossible |
| Any transformer / attention | O(n²); no int8 transformer kernels for MCU |
| On-device gradient updates | Inference only — no training infrastructure |
| Heap allocations on hot path | FreeRTOS heap fragmentation kills determinism |

---

## 2. Current System Snapshot

| Aspect | Current State |
|--------|--------------|
| Channels | 4 total; ch0–ch2 forearm (FCR, FCU, extensor), ch3 bicep (excluded from hand classifier) |
| Sampling | 1000 Hz, timer/polling (jitter — fix with Change A) |
| Window | 150 samples (150ms), 25-sample hop (25ms) |
| Features | 12: RMS, WL, ZC, SSC × 3 channels |
| Classifier | Single LDA, float32 weights in C header |
| Label alignment | RMS onset detection — missing +100ms forward shift (Change 0) |
| Normalization | Per-session z-score in Python; no on-device equivalent (Change D) |
| Smoothing | EMA (α=0.7) + majority vote (5) + debounce (3 counts) |
| Confidence rejection | None — always outputs a class (Change C) |
| Signal filtering | Analogue only via MyoWare (Change B adds software IIR) |
| Gestures | 5: fist, hook\_em, open, rest, thumbs\_up |
| Training data | 15 HDF5 sessions, 1 user |

---

## 2.1 — Confirmed Firmware Architecture (From Codebase Exploration)

> Confirmed by direct codebase inspection 2026-02-24. All file paths relative to
> `C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/`

### ADC Pin Mapping (`drivers/emg_sensor.c`)

| Channel | ADC Channel | GPIO | Muscle Location | Role in Classifier |
|---------|-------------|------|-----------------|-------------------|
| ch0 | `ADC_CHANNEL_1` | GPIO 2 | Forearm Belly (FCR) | Primary flexion signal |
| ch1 | `ADC_CHANNEL_2` | GPIO 3 | Forearm Extensors | Extension signal |
| ch2 | `ADC_CHANNEL_8` | GPIO 9 | Forearm Contractors (FCU) | Ulnar flexion signal |
| ch3 | `ADC_CHANNEL_9` | GPIO 10 | Bicep | Independent — see Section 2.2 |

**Current ADC driver**: `adc_oneshot` (polling — **NOT DMA continuous yet**; Change A migrates this)
- Attenuation: `ADC_ATTEN_DB_12` (0–3.9V full-scale range)
- Calibration: `adc_cali_curve_fitting` scheme
- Output: calibrated millivolts as `uint16_t` packed into `emg_sample_t.channels[4]`
- Timing: `vTaskDelay(1)` in `run_inference_loop()` provides the ~1ms sample interval

### Current Task Structure (`app/main.c`)

| Task | Priority | Stack | Core Pinning | Role |
|------|----------|-------|--------------|------|
| `app_main` (implicit) | Default | Default | None | Runs inference loop + state machine |
| `serial_input_task` | 5 | 4096 B | **None** | Parses UART JSON commands |

**No other tasks exist.** Change A will add `adc_sampling_task` pinned to Core 0.
The inference loop runs on `app_main`'s default task — no explicit core affinity.

### State Machine (`app/main.c`)

```
STATE_IDLE  ─(BLE/UART connect)─►  STATE_CONNECTED
                                         │
                   {"cmd": "start_stream"}▼
                                  STATE_STREAMING    (sends raw ADC over UART for Python)
                                         │
                  {"cmd": "start_predict"}▼
                                  STATE_PREDICTING   (runs run_inference_loop())
```
Communication: UART at 921600 baud, JSON framing.

### Complete Data Flow (Exact Function Names)

```
emg_sensor_read(&sample)
  │  drivers/emg_sensor.c
  │  adc_oneshot_read() × 4 channels → adc_cali_raw_to_voltage() → uint16_t mV
  │  Result: sample.channels[4] = {ch0_mV, ch1_mV, ch2_mV, ch3_mV}
  │
  ▼  Called every ~1ms (vTaskDelay(1) in run_inference_loop)
inference_add_sample(sample.channels)
  │  core/inference.c
  │  Writes to circular window_buffer[150][4]
  │  Returns true when buffer is full (after first 150 samples)
  │
  ▼  Called every 25 samples (stride_counter % INFERENCE_HOP_SIZE == 0)
inference_predict(&confidence)
  │  core/inference.c
  │  compute_features() → LDA scores → softmax → EMA → majority vote → debounce
  │  Returns: gesture class index (int), fills confidence (float)
  │
  ▼
inference_get_gesture_enum(class_idx)
  │  core/inference.c
  │  String match on MODEL_CLASS_NAMES[] → gesture_t enum value
  │
  ▼
gestures_execute(gesture)
     core/gestures.c
     switch(gesture) → servo PWM via LEDC driver
     Servo pins: GPIO 1,4,5,6,7 (Thumb, Index, Middle, Ring, Pinky)
```

### Current Buffer State

```c
// core/inference.c line 19:
static uint16_t window_buffer[INFERENCE_WINDOW_SIZE][NUM_CHANNELS];
//       ^^^^^^^^ MUST change to float when adding IIR filter (Change B)
//
// uint16_t: 150 × 4 × 2 = 1,200 bytes in internal SRAM
// float:    150 × 4 × 4 = 2,400 bytes in internal SRAM  (still trivially small)
//
// Reason for change: IIR filter outputs float; casting back to uint16_t loses
// sub-mV precision and re-introduces the quantization noise we just filtered out.
```

### `platformio.ini` Current State (`EMG_Arm/platformio.ini`)

**Current `lib_deps`**: **None** — completely empty, no external library dependencies.

Required additions per change tier:

| Change | Library | `platformio.ini` `lib_deps` entry |
|--------|---------|----------------------------------|
| B (IIR biquad) | esp-dsp | `espressif/esp-dsp @ ^2.0.0` |
| 1 (FFT features) | esp-dsp | (same — add once for both B and 1) |
| E (int8 MLP) | TFLite Micro | `tensorflow/tflite-micro` |
| F (ensemble) | esp-dsp | (same as B) |

Add to `platformio.ini` under `[env:esp32-s3-devkitc1-n16r16]`:
```ini
lib_deps =
    espressif/esp-dsp @ ^2.0.0
    ; tensorflow/tflite-micro   ← add this only when implementing Change E
```

---

## 2.2 — Bicep Channel Subsystem (ch3 / ADC_CHANNEL_9 / GPIO 10)

### Current Status

The bicep channel is:
- **Sampled**: `emg_sensor_read()` reads all 4 channels; `sample.channels[3]` holds bicep data
- **Excluded from hand classifier**: `HAND_NUM_CHANNELS = 3`; `compute_features()` explicitly
  loops `ch = 0` to `ch < HAND_NUM_CHANNELS` (i.e., ch0, ch1, ch2 only)
- **Not yet independently processed**: the comment in `inference.c` line 68
  (`"ch3 (bicep) is excluded — it will be processed independently"`) is aspirational —
  the independent processing is not yet implemented

### Phase 1 — Binary Flex/Unflex (Current Target)

Implement a simple RMS threshold detector as a new subsystem:

**New files:**
```
EMG_Arm/src/core/bicep.h
EMG_Arm/src/core/bicep.c
```

**bicep.h:**
```c
#pragma once
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    BICEP_STATE_REST = 0,
    BICEP_STATE_FLEX = 1,
} bicep_state_t;

// Call once at session start with ~3s of relaxed bicep data.
// Returns the computed threshold (also stored internally).
float bicep_calibrate(const uint16_t *ch3_samples, int n_samples);

// Call every 25ms (same hop as hand gesture inference).
// Computes RMS on the last BICEP_WINDOW_SAMPLES from the ch3 circular buffer.
bicep_state_t bicep_detect(void);

// Load/save threshold to NVS (reuse calibration.c infrastructure from Change D)
bool bicep_save_threshold(float threshold_mv);
bool bicep_load_threshold(float *threshold_mv_out);
```

**Core logic (`bicep.c`):**
```c
#define BICEP_WINDOW_SAMPLES  50     // 50ms window at 1000Hz
#define BICEP_FLEX_MULTIPLIER 2.5f   // threshold = rest_rms × 2.5
#define BICEP_HYSTERESIS      1.3f   // prevents rapid toggling at threshold boundary

static float s_threshold_mv = 0.0f;
static bicep_state_t s_state = BICEP_STATE_REST;

float bicep_calibrate(const uint16_t *ch3_samples, int n_samples) {
    float rms_sq = 0.0f;
    for (int i = 0; i < n_samples; i++)
        rms_sq += (float)ch3_samples[i] * ch3_samples[i];
    float rest_rms = sqrtf(rms_sq / n_samples);
    s_threshold_mv = rest_rms * BICEP_FLEX_MULTIPLIER;
    printf("[Bicep] Calibrated: rest_rms=%.1f mV, threshold=%.1f mV\n",
           rest_rms, s_threshold_mv);
    return s_threshold_mv;
}

bicep_state_t bicep_detect(void) {
    // Compute RMS on last BICEP_WINDOW_SAMPLES from ch3 circular buffer
    // (ch3 values are stored in window_buffer[][3] alongside hand channels)
    float rms_sq = 0.0f;
    int idx = buffer_head;
    for (int i = 0; i < BICEP_WINDOW_SAMPLES; i++) {
        float v = (float)window_buffer[idx][3];  // ch3 = bicep
        rms_sq += v * v;
        idx = (idx + 1) % INFERENCE_WINDOW_SIZE;
    }
    float rms = sqrtf(rms_sq / BICEP_WINDOW_SAMPLES);

    // Hysteresis: require FLEX_MULTIPLIER to enter flex, 1.0× to exit
    if (s_state == BICEP_STATE_REST && rms > s_threshold_mv * BICEP_HYSTERESIS)
        s_state = BICEP_STATE_FLEX;
    else if (s_state == BICEP_STATE_FLEX && rms < s_threshold_mv)
        s_state = BICEP_STATE_REST;

    return s_state;
}
```

**Integration in `main.c` `run_inference_loop()`:**
```c
// Call alongside inference_predict() every 25ms:
if (stride_counter % INFERENCE_HOP_SIZE == 0) {
    float confidence;
    int class_idx     = inference_predict(&confidence);
    gesture_t gesture = inference_get_gesture_enum(class_idx);
    bicep_state_t bicep = bicep_detect();

    // Combined actuation: hand gesture + bicep state
    // Example: bicep flex can enable/disable certain gestures,
    // or control a separate elbow/wrist joint.
    gestures_execute(gesture);
    // bicep_actuate(bicep);  ← add when elbow motor is wired
}
```

**Calibration trigger (add to serial_input_task command parsing):**
```c
// {"cmd": "calibrate_bicep"}  → collect 3s of rest data, call bicep_calibrate()
```

### Phase 2 — Continuous Angle/Velocity Prediction (Future)

When ready to move beyond binary flex/unflex:

1. **Collect angle-labeled data**: hold arm at 0°, 15°, 30°, 45°, 60°, 75°, 90°;
   log RMS at each; collect 5+ reps per angle.
2. **Fit polynomial**: `angle = a0 + a1*rms + a2*rms²` (degree-2 usually sufficient);
   use `numpy.polyfit(rms_values, angles, deg=2)`.
3. **Store coefficients in NVS**: 3 floats via `nvs_set_blob()`.
4. **On-device evaluation**: `angle = a0 + rms*(a1 + rms*a2)` — 2 MACs per inference.
5. **Velocity**: `velocity = (angle_now - angle_prev) / HOP_MS` with low-pass smoothing.

### Including ch3 in Hand Gesture Classifier (for Wrist Rotation)

If/when wrist rotation or supination gestures are added:
```python
# learning_data_collection.py — change this constant:
HAND_CHANNELS = [0, 1, 2, 3]  # was [0, 1, 2]; include bicep for rotation gestures
```
Feature count becomes: 4 channels × 20 per-ch + 10 cross-ch covariances + 6 correlations = **96 total**.
The bicep subsystem is then retired and ch3 becomes part of the main gesture classifier.

---

## 3. What Meta Built — Filtered for ESP32

Meta's Nature 2025 paper (doi:10.1038/s41586-025-09255-w) describes a 16-channel wristband
running Conv1D(16→512)+3×LSTM(512). **That exact model is not portable to ESP32-S3** (~4 MB
weights). What IS transferable:

| Meta Technique | Transferability | Where Used |
|----------------|-----------------|-----------|
| +100ms forward label shift after onset detection | ✓ Direct copy | Change 0 |
| Frequency features > amplitude features (Extended Data Fig. 6) | ✓ Core insight | Change 1, Change 6 |
| Deliberate electrode repositioning between sessions | ✓ Protocol | Change 2 |
| Window jitter + amplitude augmentation | ✓ Training | Change 3 |
| Reinhard compression `64x/(32+|x|)` | ✓ Optional flag | Change 4 |
| EMA α=0.7, threshold=0.35, debounce=50ms | ✓ Already implemented | Change C |
| Specialist features → meta-learner stacking | ✓ Adapted | Change 7 + F |
| Conv1D+LSTM architecture | ✗ Too large | Not implementable |
| Full MPF with matrix logarithm | ✗ Eigendecomp too costly | Not implementable |

---

## 4. Current Code State + Known Bugs

**All Python changes**: `C:/VSCode/Marvel_Projects/Bucky_Arm/learning_data_collection.py`
**Firmware**: `C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/inference.c`
**Config**: `C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/config/config.h`
**Weights**: `C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/model_weights.h`

### Key Symbol Locations

| Symbol | Line | Notes |
|--------|------|-------|
| Constants block | 49–94 | `NUM_CHANNELS`, `SAMPLING_RATE_HZ`, `WINDOW_SIZE_MS`, etc. |
| `align_labels_with_onset()` | 442 | RMS onset detection |
| `filter_transition_windows()` | 529 | Removes onset/offset ambiguity windows |
| `SessionStorage.save_session()` | 643 | Calls onset alignment, saves HDF5 |
| `SessionStorage.load_all_for_training()` | 871 | Returns 6 values (see bug below) |
| `EMGFeatureExtractor` class | 1404 | Current: RMS, WL, ZC, SSC only |
| `extract_features_single_channel()` | 1448 | Per-channel feature dict |
| `extract_features_window()` | 1482 | Flat array + cross-channel |
| `extract_features_batch()` | 1520 | Batch wrapper |
| `get_feature_names()` | 1545 | String names for features |
| `CalibrationTransform` class | 1562 | z-score at Python-side inference |
| `EMGClassifier` class | 1713 | LDA/QDA wrapper |
| `EMGClassifier.__init__()` | 1722 | Creates `EMGFeatureExtractor` |
| `EMGClassifier.train()` | 1735 | Feature extraction + model fit |
| `EMGClassifier._apply_session_normalization()` | 1774 | Per-session z-score |
| `EMGClassifier.cross_validate()` | 1822 | GroupKFold, trial-level |
| `EMGClassifier.export_to_header()` | 1956 | Writes `model_weights.h` |
| `EMGClassifier.save()` | 1910 | Persists model params |
| `EMGClassifier.load()` | 2089 | Reconstructs from saved params |
| `run_training_demo()` | 2333 | Main training entry point |
| `inference.c` `compute_features()` | 68 | C feature extraction |
| `inference.c` `inference_predict()` | 158 | C LDA + smoothing pipeline |

### Pending Cleanups (Do Before Any Other Code Changes)

| Item | File | Action |
|------|------|--------|
| Remove `system_mode_t` | `config/config.h` lines 93–100 | Delete the unused typedef (see Part 0, Section 0.7) |
| Add `EMG_STANDALONE` to enum | `config/config.h` line 19 | Add value to the existing MAIN_MODE enum |
| Add `STATE_LAPTOP_PREDICT` + `CMD_START_LAPTOP_PREDICT` | `app/main.c` | See Part 0, Section 0.5 for exact diffs |
| Add `run_standalone_loop()` | `app/main.c` | New function — see Part 0, Section 0.4 |
| Add `run_laptop_predict_loop()` | `app/main.c` | New function — see Part 0, Section 0.5 |
| Add `inference_get_gesture_by_name()` | `core/inference.c` + `core/inference.h` | Small helper — extracts existing strcmp logic |

### Known Bug — Line 2382

```python
# BUG: load_all_for_training() returns 6 values; this call unpacks only 5.
# session_indices_combined is silently dropped — breaks per-session normalization.
X, y, trial_ids, label_names, loaded_sessions = storage.load_all_for_training()

# FIX (apply with Change 1):
X, y, trial_ids, session_indices, label_names, loaded_sessions = storage.load_all_for_training()
```

### Current `model_weights.h` State (as of 2026-02-14 training run)

| Constant | Value | Note |
|----------|-------|------|
| `MODEL_NUM_CLASSES` | 5 | fist, hook_em, open, rest, thumbs_up |
| `MODEL_NUM_FEATURES` | 12 | RMS, WL, ZC, SSC × 3 forearm channels |
| `MODEL_CLASS_NAMES` | `{"fist","hook_em","open","rest","thumbs_up"}` | Alphabetical order |
| `MODEL_NORMALIZE_FEATURES` | *not defined yet* | Add when enabling cross-ch norm (Change B) |
| `MODEL_USE_REINHARD` | *not defined yet* | Add when enabling Reinhard compression (Change 4) |
| `FEAT_ZC_THRESH` | `0.1f` | Fraction of RMS for zero-crossing threshold |
| `FEAT_SSC_THRESH` | `0.1f` | Fraction of RMS for slope sign change threshold |

The LDA_WEIGHTS and LDA_INTERCEPTS arrays are current trained values — do not modify manually.
They are regenerated by `EMGClassifier.export_to_header()` after each training run.

### Current Feature Vector (12 features — firmware contract)

```
ch0: [0]=rms  [1]=wl  [2]=zc  [3]=ssc
ch1: [4]=rms  [5]=wl  [6]=zc  [7]=ssc
ch2: [8]=rms  [9]=wl [10]=zc [11]=ssc
```

### Target Feature Vector (69 features after Change 1)

```
Per channel (×3 channels, 20 features each):
  [0] rms  [1] wl   [2] zc   [3] ssc   [4] mav   [5] var
  [6] iemg [7] wamp [8] ar1  [9] ar2  [10] ar3  [11] ar4
 [12] mnf [13] mdf  [14] pkf [15] mnp  [16] bp0  [17] bp1
 [18] bp2 [19] bp3

ch0: indices  0–19
ch1: indices 20–39
ch2: indices 40–59

Cross-channel (9 features):
 [60] cov_ch0_ch0  [61] cov_ch0_ch1  [62] cov_ch0_ch2
 [63] cov_ch1_ch1  [64] cov_ch1_ch2  [65] cov_ch2_ch2
 [66] cor_ch0_ch1  [67] cor_ch0_ch2  [68] cor_ch1_ch2
```

### Specialist Feature Subset Indices (for Change F + Change 7)

```
TD (time-domain, 36 feat): indices [0–11, 20–31, 40–51]
FD (frequency-domain, 24 feat): indices [12–19, 32–39, 52–59]
CC (cross-channel, 9 feat): indices [60–68]
```

---

# PART II — TARGET ARCHITECTURE

## 5. Full Recommended Multi-Model Stack

```
ADC (DMA, Change A)
  └── IIR Biquad filter per channel (Change B)
        └── 150-sample circular window buffer
              │
              ▼  [every 25ms]
        compute_features()  →  69-feature vector
              │
              ▼
        calibration_apply()  (Change D — NVS z-score)
              │
              ├─── Stage 1: Activity Gate ──────────────────────────────────┐
              │    total_rms < REST_THRESHOLD?  →  return GESTURE_REST      │
              │    (skips all inference during obvious idle)                 │
              │                                                              │
              ▼  (only reached when gesture is active)                      │
        Stage 2: Parallel Specialist LDAs (Change F)                        │
              ├── LDA_TD  [TD features, 36-dim]  →  prob_td[5]             │
              ├── LDA_FD  [FD features, 24-dim]  →  prob_fd[5]             │
              └── LDA_CC  [CC features,  9-dim]  →  prob_cc[5]             │
                                                                            │
              ▼                                                             │
        Stage 3: Meta-LDA stacker (Change F)                               │
              input: [prob_td | prob_fd | prob_cc]  (15-dim)               │
              output: meta_probs[5]                                         │
                                                                            │
              ▼                                                             │
        EMA smoothing (α=0.7) on meta_probs                                │
              │                                                             │
              ├── max smoothed prob ≥ 0.50? ────── Yes ──────────────────┐ │
              │                                                           │ │
              └── No: Stage 4 Confidence Cascade (Change E)              │ │
                    run int8 MLP on full 69-feat vector                  │ │
                    use higher-confidence winner                         │ │
                          │                                              │ │
                          └────────────────────────────────────────────►│ │
                                                                         │ │
              ◄────────────────────────────────────────────────────────── │ │
              │                                                            ◄─┘
              ▼
        Stage 5: Confidence rejection (Change C)
              max_prob < 0.40?  →  return current_output (hold / GESTURE_NONE)
              │
              ▼
        Majority vote (window=5) + Debounce (count=3)
              │
              ▼
        final gesture → actuation
```

### Model Weight Footprint

| Model | Input Dim | Weights | Memory (float32) |
|-------|-----------|---------|-----------------|
| LDA_TD | 36 | 5×36 = 180 | 720 B |
| LDA_FD | 24 | 5×24 = 120 | 480 B |
| LDA_CC | 9 | 5×9 = 45 | 180 B |
| Meta-LDA | 15 | 5×15 = 75 | 300 B |
| int8 MLP [69→32→16→5] | 69 | ~2,900 | ~2.9 KB int8 |
| **Total** | | | **~4.6 KB** |

All model weights fit comfortably in internal SRAM.

---

## 6. Compute Budget for Full Stack

| Stage | Cost | Cumulative |
|-------|------|-----------|
| Feature extraction (69 feat, 128-pt FFT ×3) | 1,200 µs | 1,200 µs |
| NVS calibration apply | 10 µs | 1,210 µs |
| Activity gate (RMS check) | 5 µs | 1,215 µs |
| LDA_TD (36 feat × 5 classes) | 50 µs | 1,265 µs |
| LDA_FD (24 feat × 5 classes) | 35 µs | 1,300 µs |
| LDA_CC (9 feat × 5 classes) | 15 µs | 1,315 µs |
| Meta-LDA (15 feat × 5 classes) | 10 µs | 1,325 µs |
| EMA + confidence check | 10 µs | 1,335 µs |
| int8 MLP (worst case, ~30% of hops) | 250 µs | 1,585 µs |
| Vote + debounce | 20 µs | 1,605 µs |
| **Worst-case total** | **1,760 µs** | **7% of 25ms budget** |

---

## 7. Why This Architecture Works for 3-Channel EMG

Three channels means limited spatial information. The ensemble compensates by extracting
**maximum diversity from the temporal and spectral dimensions**:

- **LDA_TD** specializes in muscle activation *intensity and dynamics* (how hard and fast is each muscle firing)
- **LDA_FD** specializes in muscle activation *frequency content* (motor unit recruitment patterns — slow vs. fast twitch fibres fire at different frequencies)
- **LDA_CC** specializes in *inter-muscle coordination* (which muscles co-activate — the spatial "fingerprint" of each gesture)

These three signal aspects are partially uncorrelated. A gesture that confuses LDA_TD (similar amplitude patterns) may be distinguishable by LDA_FD (different frequency recruitment) or LDA_CC (different co-activation pattern). The meta-LDA learns which specialist to trust for each gesture boundary.

The int8 MLP fallback handles the residual nonlinear cases: gesture pairs where the decision boundary is curved in feature space, which LDA (linear boundary only) cannot resolve.

---

# PART III — GESTURE EXTENSIBILITY

## 8. What Changes When Adding or Removing a Gesture

The system is designed for extensibility. Adding a gesture requires **3 firmware lines and a retrain**.

### What Changes Automatically (No Manual Code Edits)

| Component | How it adapts |
|-----------|--------------|
| `MODEL_NUM_CLASSES` in `model_weights.h` | Auto-computed from training data label count |
| LDA weight array dimensions | `[MODEL_NUM_CLASSES][MODEL_NUM_FEATURES]` — regenerated by `export_to_header()` |
| `MODEL_CLASS_NAMES` array | Regenerated by `export_to_header()` |
| All ensemble LDA weight arrays | Regenerated by `export_ensemble_header()` (Change 7) |
| int8 MLP output layer | Retrained with new class count; re-exported to TFLite |
| Meta-LDA input/output dims | `META_NUM_INPUTS = 3 × MODEL_NUM_CLASSES` — auto from Python |

### What Requires Manual Code Changes

**Python side** (`learning_data_collection.py`):
```python
# 1. Add gesture name to the gesture list (1 line)
# Find where GESTURES or similar list is defined (near constants block ~line 49)
GESTURES = ['fist', 'hook_em', 'open', 'rest', 'thumbs_up', 'wrist_flex']  # example
```

**Firmware — `config.h`** (1 line per gesture):
```c
// Add enum value
typedef enum {
    GESTURE_NONE     = 0,
    GESTURE_REST     = 1,
    GESTURE_FIST     = 2,
    GESTURE_OPEN     = 3,
    GESTURE_HOOK_EM  = 4,
    GESTURE_THUMBS_UP = 5,
    GESTURE_WRIST_FLEX = 6,  // ← add this line
} gesture_t;
```

**Firmware — `inference.c`** `inference_get_gesture_enum()` (2–3 lines per gesture):
```c
if (strcmp(name, "wrist_flex") == 0 || strcmp(name, "WRIST_FLEX") == 0)
    return GESTURE_WRIST_FLEX;
```

**Firmware — `gestures.c`** (2 changes — these are easy to miss):
```c
// 1. Add to gesture_names[] static array — index MUST match gesture_t enum value:
static const char *gesture_names[GESTURE_COUNT] = {
    "NONE",       // GESTURE_NONE = 0
    "REST",       // GESTURE_REST = 1
    "FIST",       // GESTURE_FIST = 2
    "OPEN",       // GESTURE_OPEN = 3
    "HOOK_EM",    // GESTURE_HOOK_EM = 4
    "THUMBS_UP",  // GESTURE_THUMBS_UP = 5
    "WRIST_FLEX", // GESTURE_WRIST_FLEX = 6  ← add here
};

// 2. Add case to gestures_execute() switch statement:
case GESTURE_WRIST_FLEX:
    gesture_wrist_flex();   // implement the actuation function
    break;
```

**Critical**: `GESTURE_COUNT` at the end of the `gesture_t` enum in `config.h` is used as the
array size for `gesture_names[]`. It updates automatically when new enum values are added before
it. Both `gesture_names[GESTURE_COUNT]` and the switch statement must be kept in sync with
`GESTURE_COUNT`. Mismatch causes a bounds-overrun or silent misclassification.

### Complete Workflow for Adding a Gesture

```
1. Python: add gesture string to GESTURES list in learning_data_collection.py (1 line)

2. Data: collect ≥10 sessions × ≥30 reps of new gesture
   (follow Change 2 protocol: vary electrode placement between sessions)

3. Train: python learning_data_collection.py → option 3
         OR: python train_ensemble.py (after Change 7 is implemented)

4. Export: export_to_header() OR export_ensemble_header()
   → overwrites model_weights.h / model_weights_ensemble.h with new class count

5. config.h: add enum value before GESTURE_COUNT (1 line):
       GESTURE_WRIST_FLEX = 6,   // ← insert before GESTURE_COUNT
       GESTURE_COUNT             // stays last — auto-counts

6. inference.c: add string mapping in inference_get_gesture_enum() (2 lines)

7. gestures.c: add name to gesture_names[] array at correct index (1 line)

8. gestures.c: add case to gestures_execute() switch statement (3 lines)

9. Implement actuation function for new gesture (servo angles)

10. Reflash and validate: pio run -t upload
```

**Exact files touched per new gesture (summary):**
| File | What to change |
|------|---------------|
| `learning_data_collection.py` | Add string to GESTURES list |
| `config/config.h` | Add enum value before `GESTURE_COUNT` |
| `core/inference.c` | Add `strcmp` case in `inference_get_gesture_enum()` |
| `core/gestures.c` | Add to `gesture_names[]` array + add switch case |
| `core/gestures.c` | Implement `gesture_<name>()` function with servo angles |
| `core/model_weights.h` | Auto-generated — do not edit manually |

### Removing a Gesture

Removing is the same process in reverse, with one additional step: filter the HDF5 training
data to exclude sessions that contain the removed gesture's label. The simplest approach is
to pass a label whitelist to `load_all_for_training()`:

```python
# Proposed addition to load_all_for_training() — add include_labels parameter
X, y, trial_ids, session_indices, label_names, sessions = \
    storage.load_all_for_training(include_labels=['fist', 'open', 'rest', 'thumbs_up'])
    # hook_em removed — existing session files are not modified
```

---

## 9. Practical Limits of 3-Channel EMG

This is the most important constraint for gesture count:

| Gesture Count | Expected Accuracy | Notes |
|--------------|-------------------|-------|
| 3–5 gestures | >90% achievable | Current baseline target |
| 6–8 gestures | 80–90% achievable | Requires richer features + ensemble |
| 9–12 gestures | 65–80% achievable | Diminishing returns; some pairs will be confused |
| 13+ gestures | <65% | Surface EMG with 3 channels cannot reliably separate this many |

**Why 3 channels limits gesture count**: Surface EMG captures the summed electrical activity of
many motor units under each electrode. With only 3 spatial locations, gestures that recruit
overlapping muscle groups (e.g., all finger-flexion gestures recruit FCR) produce similar
signals. The frequency and coordination features from Change 1 help, but there's a hard
information-theoretic limit imposed by channel count.

**Rule of thumb**: aim for ≤8 gestures with the current 3-channel setup. For more, add the
bicep channel (ch3, currently excluded) to get 4 channels — see Section 10.

---

## 10. Specific Gesture Considerations

### Wrist Flexion / Extension
- **Feasibility**: High — FCR (ch0) activates strongly for flexion; extensor group (ch2) for extension
- **Differentiation from finger gestures**: frequency content differs (wrist involves slower motor units)
- **Recommendation**: Add these before wrist rotation — more reliable with surface EMG

### Wrist Rotation (Supination / Pronation)
- **Feasibility**: Medium — the primary supinator is a deep muscle; surface electrodes capture it weakly
- **Key helper**: the bicep activates strongly during supination → **include ch3** (`HAND_CHANNELS = [0, 1, 2, 3]`)
- **Code change for 4 channels**: Python: `HAND_CHANNELS = [0, 1, 2, 3]`; firmware: `HAND_NUM_CHANNELS` auto-updates from the exported header since `MODEL_NUM_FEATURES` is recalculated
- **Caveat**: pronation vs. rest may be harder to distinguish than supination vs. rest

### Pinch / Precision Grasp
- **Feasibility**: Medium — involves intrinsic hand muscles poorly captured by forearm electrodes
- Likely confused with open hand depending on electrode placement
- Collect with careful placement; validate cross-session accuracy before relying on it

### Including ch3 (Bicep) for Wrist Gestures

To include the bicep channel in the hand gesture classifier:
```python
# learning_data_collection.py — change this constant
HAND_CHANNELS = [0, 1, 2, 3]  # was [0, 1, 2] — add bicep channel
```
Feature count: 4 channels × 20 per-channel features + 10 cross-channel covariances + 6 correlations = **96 total features**.
The ensemble architecture handles this automatically — specialist LDA weight dimensions
recalculate at training time.

---

# PART IV — CHANGE REFERENCE

## 11. Change Classification Matrix

| Change | Category | Priority | Files | ESP32 Reflash? | Retrain? | Risk |
|--------|----------|----------|-------|----------------|----------|------|
| **C** | Firmware | **Tier 1** | inference.c | ✓ | No | **Very Low** |
| **B** | Firmware | **Tier 1** | inference.c / filter.c | ✓ | No | Low |
| **A** | Firmware | **Tier 1** | adc_sampling.c | ✓ | No | Medium |
| **0** | Python | **Tier 1** | learning_data_collection.py | No | ✓ | Low |
| **1** | Python+C | **Tier 2** | learning_data_collection.py + inference.c | ✓ after | ✓ | Medium |
| **D** | Firmware | **Tier 2** | calibration.c/.h | ✓ | No | Medium |
| **2** | Protocol | **Tier 2** | None | No | ✓ new data | None |
| **3** | Python | **Tier 2** | learning_data_collection.py | No | ✓ | Low |
| **E** | Python+FW | **Tier 3** | train_mlp_tflite.py + firmware | ✓ | ✓ | High |
| **4** | Python+C | **Tier 3** | learning_data_collection.py + inference.c | ✓ if enabled | ✓ | Low |
| **5** | Python | **Tier 3** | learning_data_collection.py | No | No | None |
| **6** | Python | **Tier 3** | learning_data_collection.py | No | ✓ | Low |
| **7** | Python | **Tier 3** | new: train_ensemble.py | No | ✓ | Medium |
| **F** | Firmware | **Tier 3** | new: inference_ensemble.c | ✓ | No (needs 7 first) | Medium |

**Recommended implementation order**: C → B → A → 0 → 1 → D → 2 → 3 → 5 (benchmark) → 7+F → E

---

# PART V — FIRMWARE CHANGES

## Change A — DMA-Driven ADC Sampling (Migration from `adc_oneshot` to `adc_continuous`)

**Priority**: Tier 1
**Current driver**: `adc_oneshot_read()` polling in `drivers/emg_sensor.c`. Timing is
controlled by `vTaskDelay(1)` in `run_inference_loop()` — subject to FreeRTOS scheduler
jitter of ±0.5–1ms, which corrupts frequency-domain features and ADC burst grouping.
**Why**: `adc_continuous` runs entirely in hardware DMA. Sample-to-sample jitter drops from
±1ms to <10µs. CPU overhead between samples is zero. Required for frequency features (Change 1).
**Effort**: 2–4 hours (replace `emg_sensor_read()` internals; keep public API the same)

### ESP-IDF ADC Continuous API

```c
// --- Initialize (call once at startup) ---
adc_continuous_handle_t adc_handle = NULL;
adc_continuous_handle_cfg_t adc_cfg = {
    .max_store_buf_size = 4096,    // PSRAM ring buffer size (bytes)
    .conv_frame_size    = 256,     // bytes per conversion frame
};
adc_continuous_new_handle(&adc_cfg, &adc_handle);

// Actual hardware channel mapping (from emg_sensor.c):
// ch0 = ADC_CHANNEL_1 / GPIO 2  (Forearm Belly / FCR)
// ch1 = ADC_CHANNEL_2 / GPIO 3  (Forearm Extensors)
// ch2 = ADC_CHANNEL_8 / GPIO 9  (Forearm Contractors / FCU)
// ch3 = ADC_CHANNEL_9 / GPIO 10 (Bicep — independent subsystem)
adc_digi_pattern_config_t chan_cfg[4] = {
    {.atten = ADC_ATTEN_DB_12, .channel = ADC_CHANNEL_1, .unit = ADC_UNIT_1, .bit_width = ADC_BITWIDTH_12},
    {.atten = ADC_ATTEN_DB_12, .channel = ADC_CHANNEL_2, .unit = ADC_UNIT_1, .bit_width = ADC_BITWIDTH_12},
    {.atten = ADC_ATTEN_DB_12, .channel = ADC_CHANNEL_8, .unit = ADC_UNIT_1, .bit_width = ADC_BITWIDTH_12},
    {.atten = ADC_ATTEN_DB_12, .channel = ADC_CHANNEL_9, .unit = ADC_UNIT_1, .bit_width = ADC_BITWIDTH_12},
};
adc_continuous_config_t cont_cfg = {
    .sample_freq_hz = 4000,        // 4 channels × 1000 Hz = 4000 total samples/sec
    .conv_mode      = ADC_CONV_SINGLE_UNIT_1,
    .format         = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
    .pattern_num    = 4,
    .adc_pattern    = chan_cfg,
};
adc_continuous_config(adc_handle, &cont_cfg);

// --- ISR callback (fires each frame) ---
static SemaphoreHandle_t s_adc_sem;
static bool IRAM_ATTR adc_conv_done_cb(
        adc_continuous_handle_t handle,
        const adc_continuous_evt_data_t *edata, void *user_data) {
    BaseType_t hp_woken = pdFALSE;
    xSemaphoreGiveFromISR(s_adc_sem, &hp_woken);
    return hp_woken == pdTRUE;
}
adc_continuous_evt_cbs_t cbs = { .on_conv_done = adc_conv_done_cb };
adc_continuous_register_event_callbacks(adc_handle, &cbs, NULL);
adc_continuous_start(adc_handle);

// --- ADC calibration (apply per sample) ---
adc_cali_handle_t cali_handle;
adc_cali_curve_fitting_config_t cali_cfg = {
    .unit_id  = ADC_UNIT_1,
    .atten    = ADC_ATTEN_DB_12,   // matches ADC_ATTEN_DB_12 used in current emg_sensor.c
    .bitwidth = ADC_BITWIDTH_12,
};
adc_cali_create_scheme_curve_fitting(&cali_cfg, &cali_handle);

// --- Sampling task (pin to Core 0) ---
void adc_sampling_task(void *arg) {
    uint8_t result_buf[256];
    uint32_t out_len = 0;
    while (1) {
        xSemaphoreTake(s_adc_sem, portMAX_DELAY);
        adc_continuous_read(adc_handle, result_buf, sizeof(result_buf), &out_len, 0);
        // Parse: each entry is adc_digi_output_data_t
        // Apply adc_cali_raw_to_voltage() for each sample
        // Apply IIR filter (Change B) → post to inference ring buffer
    }
}
```

**Verify**: log consecutive sample timestamps via `esp_timer_get_time()`; spacing should be 1.0ms ± 0.05ms.

---

## Change B — IIR Biquad Bandpass Filter

**Priority**: Tier 1
**Why**: MyoWare analogue filters are not tunable. Software IIR removes powerline interference
(50/60 Hz), sub-20 Hz motion artifact, and >500 Hz noise — all of which inflate ZC, WL, and
other features computed at rest.
**Effort**: 2 hours

### Step 1 — Compute Coefficients in Python (one-time, offline)

```python
from scipy.signal import butter
import numpy as np

fs = 1000.0
sos = butter(N=2, Wn=[20.0, 500.0], btype='bandpass', fs=fs, output='sos')
# sos[i] = [b0, b1, b2, a0, a1, a2]
# esp-dsp Direct Form II convention: coeffs = [b0, b1, b2, -a1, -a2]
for i, s in enumerate(sos):
    b0, b1, b2, a0, a1, a2 = s
    print(f"Section {i}: {b0:.8f}f, {b1:.8f}f, {b2:.8f}f, {-a1:.8f}f, {-a2:.8f}f")
# Run this and paste the printed values into the C constants below
```

### Step 2 — Add to inference.c (after includes, before `// --- State ---`)

```c
#include "dsps_biquad.h"

// 2nd-order Butterworth bandpass 20–500 Hz @ 1000 Hz
// Coefficients: [b0, b1, b2, -a1, -a2] — Direct Form II, esp-dsp sign convention
// Regenerate with: scipy.signal.butter(N=2, Wn=[20,500], btype='bandpass', fs=1000, output='sos')
static const float BIQUAD_HP_COEFFS[5] = { /* paste section 0 output here */ };
static const float BIQUAD_LP_COEFFS[5] = { /* paste section 1 output here */ };

// Filter delay state: 3 channels × 2 stages × 2 delay elements = 12 floats (48 bytes)
static float biquad_hp_w[HAND_NUM_CHANNELS][2];
static float biquad_lp_w[HAND_NUM_CHANNELS][2];
```

Add to `inference_init()`:
```c
    memset(biquad_hp_w, 0, sizeof(biquad_hp_w));
    memset(biquad_lp_w, 0, sizeof(biquad_lp_w));
```

### Step 3 — Apply Per Sample (called before writing to window_buffer)

```c
// Apply to each channel before posting to the window buffer.
// Must be called IN ORDER for each sample (IIR has memory across calls).
static float IRAM_ATTR apply_bandpass(int ch, float raw) {
    float hp_out, lp_out;
    dsps_biquad_f32(&raw,   &hp_out, 1, (float *)BIQUAD_HP_COEFFS, biquad_hp_w[ch]);
    dsps_biquad_f32(&hp_out, &lp_out, 1, (float *)BIQUAD_LP_COEFFS, biquad_lp_w[ch]);
    return lp_out;
}
```

**Note**: `window_buffer` stores `uint16_t` — change to `float` when adding this filter, so
filtered values are stored directly without lossy integer round-trip.

**Verify**: log ZC count at rest before and after — filtered ZC should be substantially lower
(less spurious noise crossings).

---

## Change C — Confidence Rejection

**Priority**: Tier 1 — **implement this first, lowest risk of all changes**
**Why**: Without a rejection threshold, ambiguous EMG (rest-to-gesture transition,
mid-gesture fatigue, electrode lift) always produces a false actuation.
**Effort**: 15 minutes

### Step 1 — Add Constant (top of inference.c with other constants)

```c
#define CONFIDENCE_THRESHOLD 0.40f  // Reject when max smoothed prob < this.
                                    // Meta paper uses 0.35; 0.40 adds prosthetic safety margin.
                                    // Tune: lower to 0.35 if real gestures are being rejected.
```

### Step 2 — Insert After EMA Block in `inference_predict()` (after line 214)

```c
  // Confidence rejection: if the peak smoothed probability is below threshold,
  // hold the last confirmed output rather than outputting an uncertain prediction.
  // Prevents false actuations during gesture transitions and electrode artifacts.
  if (max_smoothed_prob < CONFIDENCE_THRESHOLD) {
    *confidence = max_smoothed_prob;
    return current_output;  // -1 (GESTURE_NONE) until first confident prediction
  }
```

**Verify**: arm at complete rest → confirm output stays at GESTURE_NONE and confidence logs
below 0.40. Deliberate fist → confidence rises above 0.40 within 1–3 inference cycles.

---

## Change D — On-Device NVS Calibration

**Priority**: Tier 2
**Why**: Python `CalibrationTransform` only runs during training. On-device NVS calibration
lets the ESP32 recalibrate z-score normalization at startup (3 seconds of REST) without
retraining — solving placement drift and day-to-day impedance variation.
**Effort**: 3–4 hours

### New Files

```
EMG_Arm/src/core/calibration.h
EMG_Arm/src/core/calibration.c
```

### calibration.h

```c
#pragma once
#include <stdbool.h>
#include "config/config.h"

#define CALIB_MAX_FEATURES 96  // supports up to 4-channel expansion

bool calibration_init(void);          // load from NVS at startup
void calibration_apply(float *feat);  // z-score in-place; no-op if not calibrated
bool calibration_update(const float X[][CALIB_MAX_FEATURES], int n_windows, int n_feat);
void calibration_reset(void);
bool calibration_is_valid(void);
```

### calibration.c

```c
#include "calibration.h"
#include "nvs_flash.h"
#include "nvs.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#define NVS_NAMESPACE "emg_calib"
#define NVS_KEY_MEAN  "feat_mean"
#define NVS_KEY_STD   "feat_std"
#define NVS_KEY_NFEAT "n_feat"
#define NVS_KEY_VALID "calib_ok"

static float s_mean[CALIB_MAX_FEATURES];
static float s_std[CALIB_MAX_FEATURES];
static int   s_n_feat = 0;
static bool  s_valid  = false;

bool calibration_init(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) return false;

    uint8_t valid = 0;
    size_t mean_sz = sizeof(s_mean), std_sz = sizeof(s_std);
    bool ok = (nvs_get_u8(h, NVS_KEY_VALID, &valid)         == ESP_OK) && (valid == 1) &&
              (nvs_get_i32(h, NVS_KEY_NFEAT, (int32_t*)&s_n_feat) == ESP_OK) &&
              (nvs_get_blob(h, NVS_KEY_MEAN, s_mean, &mean_sz) == ESP_OK) &&
              (nvs_get_blob(h, NVS_KEY_STD,  s_std,  &std_sz)  == ESP_OK);
    nvs_close(h);
    s_valid = ok;
    printf("[Calib] %s (%d features)\n", ok ? "Loaded from NVS" : "Not found — identity", s_n_feat);
    return ok;
}

void calibration_apply(float *feat) {
    if (!s_valid) return;
    for (int i = 0; i < s_n_feat; i++)
        feat[i] = (feat[i] - s_mean[i]) / s_std[i];
}

bool calibration_update(const float X[][CALIB_MAX_FEATURES], int n_windows, int n_feat) {
    if (n_windows < 10 || n_feat > CALIB_MAX_FEATURES) return false;
    s_n_feat = n_feat;
    memset(s_mean, 0, sizeof(s_mean));
    for (int w = 0; w < n_windows; w++)
        for (int f = 0; f < n_feat; f++)
            s_mean[f] += X[w][f];
    for (int f = 0; f < n_feat; f++) s_mean[f] /= n_windows;

    memset(s_std, 0, sizeof(s_std));
    for (int w = 0; w < n_windows; w++)
        for (int f = 0; f < n_feat; f++) {
            float d = X[w][f] - s_mean[f];
            s_std[f] += d * d;
        }
    for (int f = 0; f < n_feat; f++) {
        s_std[f] = sqrtf(s_std[f] / n_windows);
        if (s_std[f] < 1e-6f) s_std[f] = 1e-6f;
    }

    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) return false;
    nvs_set_blob(h, NVS_KEY_MEAN, s_mean, sizeof(s_mean));
    nvs_set_blob(h, NVS_KEY_STD,  s_std,  sizeof(s_std));
    nvs_set_i32(h,  NVS_KEY_NFEAT, n_feat);
    nvs_set_u8(h,   NVS_KEY_VALID, 1);
    nvs_commit(h);
    nvs_close(h);
    s_valid = true;
    printf("[Calib] Updated from %d REST windows, %d features\n", n_windows, n_feat);
    return true;
}
```

### Integration in inference.c

In `inference_predict()`, after `compute_features(features)`, before LDA:
```c
    calibration_apply(features);  // z-score using NVS-stored mean/std
```

### Startup Flow

```c
// In main application startup sequence:
calibration_init();  // load from NVS; no-op if not present yet

// When user triggers recalibration (button press or serial command):
// Collect ~120 REST windows (~3 seconds at 25ms hop)
// Call calibration_update(rest_feature_buffer, 120, MODEL_NUM_FEATURES)
```

---

## Change E — int8 MLP via TFLite Micro

**Priority**: Tier 3 — implement after Tier 1+2 changes and benchmark (Change 5) shows LDA plateauing
**Why**: LDA finds only linear decision boundaries. A 2-layer int8 MLP adds nonlinear
boundaries for gesture pairs that overlap in feature space.
**Effort**: 4–6 hours

### Python Training (new file: `train_mlp_tflite.py`)

```python
"""
Train int8 MLP for ESP32-S3 deployment via TFLite Micro.
Run AFTER Change 0 (label shift) + Change 1 (expanded features).
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from learning_data_collection import SessionStorage, EMGFeatureExtractor, HAND_CHANNELS

storage = SessionStorage()
X_raw, y, trial_ids, session_indices, label_names, _ = storage.load_all_for_training()

extractor = EMGFeatureExtractor(channels=HAND_CHANNELS, cross_channel=True)
X = extractor.extract_features_batch(X_raw).astype(np.float32)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

n_feat, n_cls = X.shape[1], len(np.unique(y))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_feat,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(n_cls, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=64, validation_split=0.1, verbose=1)

def representative_dataset():
    for i in range(0, len(X), 10):
        yield [X[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

out = Path('EMG_Arm/src/core/emg_model_data.cc')
with open(out, 'w') as f:
    f.write('#include "emg_model_data.h"\n')
    f.write(f'const int g_model_len = {len(tflite_model)};\n')
    f.write('const unsigned char g_model[] = {\n  ')
    f.write(', '.join(f'0x{b:02x}' for b in tflite_model))
    f.write('\n};\n')
print(f"Wrote {out} ({len(tflite_model)} bytes)")
```

### Firmware (inference_mlp.cc)

```cpp
#include "inference_mlp.h"
#include "emg_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static uint8_t tensor_arena[48 * 1024];  // 48 KB — tune down if memory is tight
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr, *output = nullptr;

void inference_mlp_init(void) {
    const tflite::Model *model = tflite::GetModel(g_model);
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddDequantize();
    static tflite::MicroInterpreter interp(model, resolver, tensor_arena, sizeof(tensor_arena));
    interpreter = &interp;
    interpreter->AllocateTensors();
    input  = interpreter->input(0);
    output = interpreter->output(0);
}

int inference_mlp_predict(const float *features, int n_feat, float *conf_out) {
    float iscale = input->params.scale;
    int   izp    = input->params.zero_point;
    for (int i = 0; i < n_feat; i++) {
        int q = (int)roundf(features[i] / iscale) + izp;
        input->data.int8[i] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
    }
    interpreter->Invoke();

    float oscale = output->params.scale;
    int   ozp    = output->params.zero_point;
    float max_p = -1e9f;
    int max_c = 0;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
        float p = (output->data.int8[c] - ozp) * oscale;
        if (p > max_p) { max_p = p; max_c = c; }
    }
    *conf_out = max_p;
    return max_c;
}
```

**platformio.ini addition**:
```ini
lib_deps =
    tensorflow/tflite-micro
```

---

## Change F — Ensemble Inference Pipeline

**Priority**: Tier 3 (requires Change 1 features + Change 7 training + Change E MLP)
**Why**: This is the full recommended architecture from Part II.
**Effort**: 3–4 hours firmware (after Python ensemble is trained and exported)

### New Files

```
EMG_Arm/src/core/inference_ensemble.c
EMG_Arm/src/core/inference_ensemble.h
EMG_Arm/src/core/model_weights_ensemble.h   (generated by Change 7 Python script)
```

### inference_ensemble.h

```c
#pragma once
#include <stdbool.h>

void inference_ensemble_init(void);
int  inference_ensemble_predict(float *confidence);
```

### inference_ensemble.c

```c
#include "inference_ensemble.h"
#include "inference.h"          // for compute_features(), calibration_apply()
#include "inference_mlp.h"      // for inference_mlp_predict()
#include "model_weights_ensemble.h"
#include "config/config.h"
#include "dsps_dotprod.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#define ENSEMBLE_EMA_ALPHA      0.70f
#define ENSEMBLE_CONF_THRESHOLD 0.50f  // below this: escalate to MLP fallback
#define REJECT_THRESHOLD        0.40f  // below this even after MLP: hold output
#define REST_ACTIVITY_THRESHOLD 0.05f  // total_rms below this → skip inference, return REST

// EMA state
static float s_smoothed[MODEL_NUM_CLASSES];
// Vote + debounce (reuse existing pattern from inference.c)
static int s_vote_history[5];
static int s_vote_head = 0;
static int s_current_output = -1;
static int s_pending_output = -1;
static int s_pending_count  = 0;

// --- Generic LDA softmax predict ---
// weights: [n_classes][n_feat], intercepts: [n_classes]
// proba_out: [n_classes] — caller-provided output
static void lda_softmax(const float *feat, int n_feat,
                         const float *weights_flat, const float *intercepts,
                         int n_classes, float *proba_out) {
    float raw[MODEL_NUM_CLASSES];
    float max_raw = -1e9f, sum_exp = 0.0f;

    for (int c = 0; c < n_classes; c++) {
        raw[c] = intercepts[c];
        // dsps_dotprod_f32 requires 4-byte aligned arrays and length multiple of 4;
        // for safety use plain loop — compiler will auto-vectorize with -O2
        const float *w = weights_flat + c * n_feat;
        for (int f = 0; f < n_feat; f++) raw[c] += feat[f] * w[f];
        if (raw[c] > max_raw) max_raw = raw[c];
    }
    for (int c = 0; c < n_classes; c++) {
        proba_out[c] = expf(raw[c] - max_raw);
        sum_exp += proba_out[c];
    }
    for (int c = 0; c < n_classes; c++) proba_out[c] /= sum_exp;
}

void inference_ensemble_init(void) {
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
        s_smoothed[c] = 1.0f / MODEL_NUM_CLASSES;
    for (int i = 0; i < 5; i++) s_vote_history[i] = -1;
    s_vote_head = 0;
    s_current_output = -1;
    s_pending_output = -1;
    s_pending_count  = 0;
}

int inference_ensemble_predict(float *confidence) {
    // 1. Extract features (shared with single-model path)
    float features[MODEL_NUM_FEATURES];
    compute_features(features);
    calibration_apply(features);

    // 2. Activity gate — skip inference during obvious REST
    float total_rms_sq = 0.0f;
    for (int ch = 0; ch < HAND_NUM_CHANNELS; ch++) {
        float r = features[ch * ENSEMBLE_PER_CH_FEATURES]; // RMS is index 0 per channel
        total_rms_sq += r * r;
    }
    if (sqrtf(total_rms_sq) < REST_ACTIVITY_THRESHOLD) {
        *confidence = 1.0f;
        return GESTURE_REST;
    }

    // 3. Specialist LDAs
    float prob_td[MODEL_NUM_CLASSES];
    float prob_fd[MODEL_NUM_CLASSES];
    float prob_cc[MODEL_NUM_CLASSES];

    lda_softmax(features + TD_FEAT_OFFSET, TD_NUM_FEATURES,
                (const float *)LDA_TD_WEIGHTS, LDA_TD_INTERCEPTS,
                MODEL_NUM_CLASSES, prob_td);
    lda_softmax(features + FD_FEAT_OFFSET, FD_NUM_FEATURES,
                (const float *)LDA_FD_WEIGHTS, LDA_FD_INTERCEPTS,
                MODEL_NUM_CLASSES, prob_fd);
    lda_softmax(features + CC_FEAT_OFFSET, CC_NUM_FEATURES,
                (const float *)LDA_CC_WEIGHTS, LDA_CC_INTERCEPTS,
                MODEL_NUM_CLASSES, prob_cc);

    // 4. Meta-LDA stacker
    float meta_in[META_NUM_INPUTS];  // = 3 * MODEL_NUM_CLASSES
    memcpy(meta_in,                        prob_td, MODEL_NUM_CLASSES * sizeof(float));
    memcpy(meta_in +   MODEL_NUM_CLASSES,  prob_fd, MODEL_NUM_CLASSES * sizeof(float));
    memcpy(meta_in + 2*MODEL_NUM_CLASSES,  prob_cc, MODEL_NUM_CLASSES * sizeof(float));

    float meta_probs[MODEL_NUM_CLASSES];
    lda_softmax(meta_in, META_NUM_INPUTS,
                (const float *)META_LDA_WEIGHTS, META_LDA_INTERCEPTS,
                MODEL_NUM_CLASSES, meta_probs);

    // 5. EMA smoothing on meta output
    float max_smooth = 0.0f;
    int winner = 0;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
        s_smoothed[c] = ENSEMBLE_EMA_ALPHA * s_smoothed[c] +
                        (1.0f - ENSEMBLE_EMA_ALPHA) * meta_probs[c];
        if (s_smoothed[c] > max_smooth) { max_smooth = s_smoothed[c]; winner = c; }
    }

    // 6. Confidence cascade: escalate to MLP if meta-LDA is uncertain
    if (max_smooth < ENSEMBLE_CONF_THRESHOLD) {
        float mlp_conf = 0.0f;
        int mlp_winner = inference_mlp_predict(features, MODEL_NUM_FEATURES, &mlp_conf);
        if (mlp_conf > max_smooth) { winner = mlp_winner; max_smooth = mlp_conf; }
    }

    // 7. Reject if still uncertain
    if (max_smooth < REJECT_THRESHOLD) {
        *confidence = max_smooth;
        return s_current_output;
    }

    *confidence = max_smooth;

    // 8. Majority vote (window = 5)
    s_vote_history[s_vote_head] = winner;
    s_vote_head = (s_vote_head + 1) % 5;
    int counts[MODEL_NUM_CLASSES] = {0};
    for (int i = 0; i < 5; i++)
        if (s_vote_history[i] >= 0) counts[s_vote_history[i]]++;
    int majority = 0, majority_cnt = 0;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
        if (counts[c] > majority_cnt) { majority_cnt = counts[c]; majority = c; }

    // 9. Debounce (3 consecutive predictions to change output)
    int final = s_current_output;
    if (s_current_output == -1) {
        s_current_output = majority; final = majority;
    } else if (majority == s_current_output) {
        s_pending_output = majority; s_pending_count = 1;
    } else if (majority == s_pending_output) {
        if (++s_pending_count >= 3) { s_current_output = majority; final = majority; }
    } else {
        s_pending_output = majority; s_pending_count = 1;
    }

    return final;
}
```

### model_weights_ensemble.h Layout (generated by Change 7)

```c
// Auto-generated by train_ensemble.py — do not edit manually
#pragma once

#define MODEL_NUM_CLASSES    5       // auto-computed from training data
#define MODEL_NUM_FEATURES   69      // total feature count (after Change 1)
#define ENSEMBLE_PER_CH_FEATURES 20  // features per channel

// Specialist feature subset offsets and sizes
#define TD_FEAT_OFFSET  0
#define TD_NUM_FEATURES 36   // time-domain: indices 0–11, 20–31, 40–51
#define FD_FEAT_OFFSET  12   // NOTE: FD features are interleaved per-channel
#define FD_NUM_FEATURES 24   // freq-domain: indices 12–19, 32–39, 52–59
#define CC_FEAT_OFFSET  60
#define CC_NUM_FEATURES 9    // cross-channel: indices 60–68

#define META_NUM_INPUTS (3 * MODEL_NUM_CLASSES)  // = 15

// Specialist LDA weights (flat row-major: [n_classes][n_feat])
extern const float LDA_TD_WEIGHTS[MODEL_NUM_CLASSES][TD_NUM_FEATURES];
extern const float LDA_TD_INTERCEPTS[MODEL_NUM_CLASSES];

extern const float LDA_FD_WEIGHTS[MODEL_NUM_CLASSES][FD_NUM_FEATURES];
extern const float LDA_FD_INTERCEPTS[MODEL_NUM_CLASSES];

extern const float LDA_CC_WEIGHTS[MODEL_NUM_CLASSES][CC_NUM_FEATURES];
extern const float LDA_CC_INTERCEPTS[MODEL_NUM_CLASSES];

// Meta-LDA weights
extern const float META_LDA_WEIGHTS[MODEL_NUM_CLASSES][META_NUM_INPUTS];
extern const float META_LDA_INTERCEPTS[MODEL_NUM_CLASSES];

// Class names (for inference_get_gesture_enum)
extern const char *MODEL_CLASS_NAMES[MODEL_NUM_CLASSES];
```

**Important note on FD features**: the frequency-domain features are interleaved at indices
[12–19] for ch0, [32–39] for ch1, [52–59] for ch2. The `lda_softmax` call for LDA_FD must
pass a **gathered** (non-contiguous) sub-vector. The cleanest approach is to gather them into
a contiguous buffer before calling lda_softmax:

```c
// Gather FD features into contiguous buffer before LDA_FD
float fd_buf[FD_NUM_FEATURES];
for (int ch = 0; ch < HAND_NUM_CHANNELS; ch++)
    memcpy(fd_buf + ch*8, features + ch*20 + 12, 8 * sizeof(float));
lda_softmax(fd_buf, FD_NUM_FEATURES, ...);
```

Similarly for TD features. This gather costs <5 µs — negligible.

---

# PART VI — PYTHON/TRAINING CHANGES

## Change 0 — Forward Label Shift

**Priority**: Tier 1
**Source**: Meta Nature 2025, Methods: "Discrete-gesture time alignment"
**Why**: +100ms shift after onset detection gives the classifier 100ms of pre-event "building"
signal, dramatically cleaning the decision boundary near gesture onset.
**ESP32 impact**: None.

### Step 1 — Add Constant After Line 94

```python
# After: TRANSITION_END_MS = 150
LABEL_FORWARD_SHIFT_MS = 100  # shift label boundaries +100ms after onset alignment
                               # Source: Kaifosh et al. Nature 2025. doi:10.1038/s41586-025-09255-w
```

### Step 2 — Apply Shift in `SessionStorage.save_session()` (after line ~704)

Find and insert after:
```python
            print(f"[Storage] Labels aligned: {changed}/{len(labels)} windows shifted")
```

Insert:
```python
        if LABEL_FORWARD_SHIFT_MS > 0:
            shift_windows = max(1, round(LABEL_FORWARD_SHIFT_MS / HOP_SIZE_MS))
            shifted = list(aligned_labels)
            for i in range(1, len(aligned_labels)):
                if aligned_labels[i] != aligned_labels[i - 1]:
                    for j in range(i, min(i + shift_windows, len(aligned_labels))):
                        if shifted[j] == aligned_labels[i]:
                            shifted[j] = aligned_labels[i - 1]
            n_shifted = sum(1 for a, b in zip(aligned_labels, shifted) if a != b)
            aligned_labels = shifted
            print(f"[Storage] Forward label shift (+{LABEL_FORWARD_SHIFT_MS}ms): {n_shifted} windows adjusted")
```

### Step 3 — Reduce TRANSITION_START_MS

```python
TRANSITION_START_MS = 200   # was 300 — reduce because 100ms shift already adds pre-event context
```

**Verify**: printout shows `N windows adjusted` where N is 5–20% of total windows per session.

---

## Change 1 — Expanded Feature Set

**Priority**: Tier 2
**Why**: 12 → 69 features; adds frequency-domain and cross-channel information that is
structurally more informative than amplitude alone (Meta Extended Data Fig. 6).
**ESP32 impact**: retrain → export new `model_weights.h`; port selected features to C.

### Sub-change 1A — Expand `extract_features_single_channel()` (line 1448)

Replace the entire function body:

```python
    def extract_features_single_channel(self, signal: np.ndarray) -> dict:
        if getattr(self, 'reinhard', False):
            signal = 64.0 * signal / (32.0 + np.abs(signal))

        signal = signal - np.mean(signal)
        N = len(signal)

        # --- Time domain ---
        rms  = np.sqrt(np.mean(signal ** 2))
        diff = np.diff(signal)
        wl   = np.sum(np.abs(diff))
        zc_thresh  = self.zc_threshold_percent * rms
        ssc_thresh = (self.ssc_threshold_percent * rms) ** 2
        sign_ch = signal[:-1] * signal[1:] < 0
        zc  = int(np.sum(sign_ch & (np.abs(diff) > zc_thresh)))
        d_l = signal[1:-1] - signal[:-2]
        d_r = signal[1:-1] - signal[2:]
        ssc = int(np.sum((d_l * d_r) > ssc_thresh))
        mav  = np.mean(np.abs(signal))
        var  = np.mean(signal ** 2)
        iemg = np.sum(np.abs(signal))
        wamp = int(np.sum(np.abs(diff) > 0.15 * rms))

        # AR(4) via Yule-Walker
        ar = np.zeros(4)
        if rms > 1e-6:
            try:
                from scipy.linalg import solve_toeplitz
                r = np.array([np.dot(signal[i:], signal[:N-i]) / N for i in range(5)])
                if r[0] > 1e-10:
                    ar = solve_toeplitz(r[:4], -r[1:5])
            except Exception:
                pass

        # --- Frequency domain (20–500 Hz) ---
        freqs = np.fft.rfftfreq(N, d=1.0 / SAMPLING_RATE_HZ)
        psd   = np.abs(np.fft.rfft(signal)) ** 2 / N
        m     = (freqs >= 20) & (freqs <= 500)
        f_m, p_m = freqs[m], psd[m]
        tp = np.sum(p_m) + 1e-10
        mnf = float(np.sum(f_m * p_m) / tp)
        cum = np.cumsum(p_m)
        mdf = float(f_m[min(np.searchsorted(cum, tp / 2), len(f_m) - 1)])
        pkf = float(f_m[np.argmax(p_m)]) if len(p_m) > 0 else 0.0
        mnp = float(tp / max(len(p_m), 1))

        # Bandpower in 4 physiological bands (mirrors firmware esp-dsp FFT bands)
        bands = [(20, 80), (80, 150), (150, 300), (300, 500)]
        bp = [float(np.sum(psd[(freqs >= lo) & (freqs < hi)])) for lo, hi in bands]

        return {
            'rms': rms, 'wl': wl, 'zc': zc, 'ssc': ssc,
            'mav': mav, 'var': var, 'iemg': iemg, 'wamp': wamp,
            'ar1': float(ar[0]), 'ar2': float(ar[1]),
            'ar3': float(ar[2]), 'ar4': float(ar[3]),
            'mnf': mnf, 'mdf': mdf, 'pkf': pkf, 'mnp': mnp,
            'bp0': bp[0], 'bp1': bp[1], 'bp2': bp[2], 'bp3': bp[3],
        }
```

### Sub-change 1B — Update `extract_features_window()` Return Block (line 1482)

Replace the return section:

```python
        FEATURE_ORDER = ['rms', 'wl', 'zc', 'ssc', 'mav', 'var', 'iemg', 'wamp',
                         'ar1', 'ar2', 'ar3', 'ar4', 'mnf', 'mdf', 'pkf', 'mnp',
                         'bp0', 'bp1', 'bp2', 'bp3']
        NORMALIZE_KEYS = {'rms', 'wl', 'mav', 'iemg'}

        features = []
        for ch_features in all_ch_features:
            for key in FEATURE_ORDER:
                val = ch_features.get(key, 0.0)
                if self.normalize and key in NORMALIZE_KEYS:
                    val = val / norm_factor
                features.append(float(val))

        if self.cross_channel and window.shape[1] >= 2:
            sel   = window[:, channel_indices].astype(np.float32)
            wc    = sel - sel.mean(axis=0)
            cov   = (wc.T @ wc) / len(wc)
            ri, ci = np.triu_indices(len(channel_indices))
            features.extend(cov[ri, ci].tolist())
            stds = np.sqrt(np.diag(cov)) + 1e-10
            cor  = cov / np.outer(stds, stds)
            ro, co = np.triu_indices(len(channel_indices), k=1)
            features.extend(cor[ro, co].tolist())

        return np.array(features, dtype=np.float32)
```

### Sub-change 1C — Update `EMGFeatureExtractor.__init__()` (line 1430)

```python
    def __init__(self, zc_threshold_percent=0.1, ssc_threshold_percent=0.1,
                 channels=None, normalize=True, cross_channel=True, reinhard=False):
        self.zc_threshold_percent  = zc_threshold_percent
        self.ssc_threshold_percent = ssc_threshold_percent
        self.channels      = channels
        self.normalize     = normalize
        self.cross_channel = cross_channel
        self.reinhard      = reinhard
```

### Sub-change 1D — Update Feature Count in `extract_features_batch()` (line 1520)

Replace `n_features = n_channels * 4`:
```python
        per_ch = 20
        if self.cross_channel and n_channels >= 2:
            n_features = n_channels * per_ch + \
                         n_channels*(n_channels+1)//2 + n_channels*(n_channels-1)//2
        else:
            n_features = n_channels * per_ch
```

### Sub-change 1E — Update `get_feature_names()` (line 1545)

```python
    def get_feature_names(self, n_channels=0):
        ch_idx = self.channels if self.channels is not None else list(range(n_channels))
        ORDER = ['rms','wl','zc','ssc','mav','var','iemg','wamp',
                 'ar1','ar2','ar3','ar4','mnf','mdf','pkf','mnp','bp0','bp1','bp2','bp3']
        names = [f'ch{ch}_{f}' for ch in ch_idx for f in ORDER]
        if self.cross_channel and len(ch_idx) >= 2:
            n = len(ch_idx)
            names += [f'cov_ch{ch_idx[i]}_ch{ch_idx[j]}' for i in range(n) for j in range(i, n)]
            names += [f'cor_ch{ch_idx[i]}_ch{ch_idx[j]}' for i in range(n) for j in range(i+1, n)]
        return names
```

### Sub-change 1F — Update `EMGClassifier.__init__()` (line 1722)

```python
        self.feature_extractor = EMGFeatureExtractor(
            channels=HAND_CHANNELS, cross_channel=True, reinhard=False)
```

### Sub-change 1G — Update `save()` (line 1910) and `load()` (line 2089)

In `save()`, add to `feature_extractor_params` dict:
```python
                'cross_channel': getattr(self.feature_extractor, 'cross_channel', True),
                'reinhard':      getattr(self.feature_extractor, 'reinhard', False),
```

In `load()`, update `EMGFeatureExtractor(...)` constructor:
```python
        classifier.feature_extractor = EMGFeatureExtractor(
            zc_threshold_percent  = params.get('zc_threshold_percent', 0.1),
            ssc_threshold_percent = params.get('ssc_threshold_percent', 0.1),
            channels              = params.get('channels', HAND_CHANNELS),
            normalize             = params.get('normalize', False),
            cross_channel         = params.get('cross_channel', True),
            reinhard              = params.get('reinhard', False),
        )
```

### Also Fix Bug at Line 2382

```python
X, y, trial_ids, session_indices, label_names, loaded_sessions = storage.load_all_for_training()
```

---

## Change 2 — Electrode Repositioning Protocol

**Protocol**: no code changes.
> *"Between sessions within a single day, the participants remove and slightly reposition the
> sEMG wristband to enable generalization across different recording positions."*
> — Meta Nature 2025 Methods

- Session 1: standard placement
- Session 2: band 1–2 cm up the forearm
- Session 3: band 1–2 cm down the forearm
- Session 4+: slight axial rotation or return to any above position

The per-session z-score normalization in `_apply_session_normalization()` handles the
resulting amplitude shifts. Perform **fast, natural** gestures — not slow/deliberate.

---

## Change 3 — Data Augmentation

**Priority**: Tier 2. Apply to **raw windows BEFORE feature extraction**.

Insert before the `# === LDA CLASSIFIER ===` comment (~line 1709):

```python
def augment_emg_batch(X, y, multiplier=3, seed=42):
    """
    Augment raw EMG windows for training robustness.
    Must be called on raw windows (n_windows, n_samples, n_channels),
    not on pre-computed features.
    Source (window jitter): Kaifosh et al. Nature 2025. doi:10.1038/s41586-025-09255-w
    """
    rng = np.random.default_rng(seed)
    aug_X, aug_y = [X], [y]
    for _ in range(multiplier - 1):
        Xc = X.copy().astype(np.float32)
        Xc *= rng.uniform(0.80, 1.20, (len(X), 1, 1)).astype(np.float32)          # amplitude
        rms = np.sqrt(np.mean(Xc**2, axis=(1,2), keepdims=True)) + 1e-8
        Xc += rng.standard_normal(Xc.shape).astype(np.float32) * (0.05 * rms)     # noise
        Xc += rng.uniform(-20., 20., (len(X), 1, X.shape[2])).astype(np.float32)  # DC jitter
        shifts = rng.integers(-5, 6, size=len(X))
        for i in range(len(Xc)):
            if shifts[i]: Xc[i] = np.roll(Xc[i], shifts[i], axis=0)              # jitter
        aug_X.append(Xc); aug_y.append(y)
    return np.concatenate(aug_X), np.concatenate(aug_y)
```

In `EMGClassifier.train()`, replace the start of the function's feature extraction block:

```python
        if getattr(self, 'use_augmentation', True):
            X_aug, y_aug = augment_emg_batch(X, y, multiplier=3)
            print(f"[Classifier] Augmented: {len(X)} → {len(X_aug)} windows")
        else:
            X_aug, y_aug = X, y
        X_features = self.feature_extractor.extract_features_batch(X_aug)
        # ... then use y_aug instead of y for model.fit()
```

---

## Change 4 — Reinhard Compression (Optional)

**Formula**: `output = 64 × x / (32 + |x|)`
**Enable in Python**: set `reinhard=True` in `EMGFeatureExtractor` constructor (Change 1F).

**Enable in firmware** (`inference.c` `compute_features()`, after signal copy loop, before mean calc):
```c
#if MODEL_USE_REINHARD
    for (int i = 0; i < INFERENCE_WINDOW_SIZE; i++) {
        float x = signal[i];
        signal[i] = 64.0f * x / (32.0f + fabsf(x));
    }
#endif
```
Add `#define MODEL_USE_REINHARD 0` to `model_weights.h` (set to `1` when Python uses `reinhard=True`).
**Python and firmware MUST match.** Mismatch silently corrupts all predictions.

---

## Change 5 — Classifier Benchmark

**Purpose**: tells you whether LDA accuracy plateau is a features problem (all classifiers similar → add features) or a model complexity problem (SVM/MLP >> LDA → implement Change E/F).

Add after `run_training_demo()`:

```python
def run_classifier_benchmark():
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

    storage = SessionStorage()
    X_raw, y, trial_ids, session_indices, label_names, _ = storage.load_all_for_training()
    extractor = EMGFeatureExtractor(channels=HAND_CHANNELS, cross_channel=True)
    X = extractor.extract_features_batch(X_raw)
    X = EMGClassifier()._apply_session_normalization(X, session_indices, y=y)

    clfs = {
        'LDA (ESP32 model)':  LinearDiscriminantAnalysis(),
        'QDA':                QuadraticDiscriminantAnalysis(reg_param=0.1),
        'SVM-RBF':            Pipeline([('s', StandardScaler()), ('m', SVC(kernel='rbf', C=10))]),
        'MLP-128-64':         Pipeline([('s', StandardScaler()),
                                         ('m', MLPClassifier(hidden_layer_sizes=(128,64),
                                                             max_iter=1000, early_stopping=True))]),
    }
    gkf = GroupKFold(n_splits=5)
    print(f"\n{'Classifier':<22} {'Mean CV':>8} {'Std':>6}")
    print("-" * 40)
    for name, clf in clfs.items():
        sc = cross_val_score(clf, X, y, cv=gkf, groups=trial_ids, scoring='accuracy')
        print(f"  {name:<20} {sc.mean()*100:>7.1f}%  ±{sc.std()*100:.1f}%")
    print("\n  → If LDA ≈ SVM: features are the bottleneck (add Change 1 features)")
    print("  → If SVM >> LDA: model complexity bottleneck (implement Change F ensemble)")
```

---

## Change 6 — Simplified MPF Features

**Python training only** — not worth porting to ESP32 directly (use bandpower bp0–bp3 from Change 1 as the firmware-side approximation).

Add after `EMGFeatureExtractor` class:

```python
class MPFFeatureExtractor:
    """
    Simplified 3-channel MPF: CSD upper triangle per 6 frequency bands = 36 features.
    Python training only. Omits matrix logarithm (not needed for 3 channels).
    Source: Kaifosh et al. Nature 2025. doi:10.1038/s41586-025-09255-w
    ESP32 approximation: use bp0–bp3 from EMGFeatureExtractor (Change 1).
    """
    BANDS = [(0,62),(62,125),(125,187),(187,250),(250,375),(375,500)]

    def __init__(self, channels=None, log_diagonal=True):
        self.channels = channels or HAND_CHANNELS
        self.log_diag = log_diagonal
        self.n_ch = len(self.channels)
        self._r, self._c = np.triu_indices(self.n_ch)
        self.n_features = len(self.BANDS) * len(self._r)

    def extract_window(self, window):
        sig   = window[:, self.channels].astype(np.float64)
        N     = len(sig)
        freqs = np.fft.rfftfreq(N, d=1.0/SAMPLING_RATE_HZ)
        Xf    = np.fft.rfft(sig, axis=0)
        feats = []
        for lo, hi in self.BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                feats.extend([0.0] * len(self._r)); continue
            CSD = (Xf[mask].conj().T @ Xf[mask]).real / N
            if self.log_diag:
                for k in range(self.n_ch): CSD[k,k] = np.log(max(CSD[k,k], 1e-10))
            feats.extend(CSD[self._r, self._c].tolist())
        return np.array(feats, dtype=np.float32)

    def extract_batch(self, X):
        out = np.zeros((len(X), self.n_features), dtype=np.float32)
        for i in range(len(X)): out[i] = self.extract_window(X[i])
        return out
```

In `EMGClassifier.train()`, after standard feature extraction:
```python
        if getattr(self, 'use_mpf', False):
            mpf = MPFFeatureExtractor(channels=HAND_CHANNELS)
            X_features = np.hstack([X_features, mpf.extract_batch(X_aug)])
```

---

## Change 7 — Ensemble Training

**Priority**: Tier 3 (implements Change F's training side)
**New file**: `C:/VSCode/Marvel_Projects/Bucky_Arm/train_ensemble.py`

```python
"""
Train the full 3-specialist-LDA + meta-LDA ensemble.
Requires Change 1 (expanded features) to be implemented first.
Exports model_weights_ensemble.h for firmware Change F.

Architecture:
  LDA_TD (36 time-domain feat) ─┐
  LDA_FD (24 freq-domain feat)  ├─ 15 probs ─► Meta-LDA ─► final class
  LDA_CC (9  cross-ch feat)     ─┘
"""
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, GroupKFold, cross_val_score
import sys
sys.path.insert(0, str(Path(__file__).parent))
from learning_data_collection import (
    SessionStorage, EMGFeatureExtractor, HAND_CHANNELS
)

# ─── Load and extract features ───────────────────────────────────────────────
storage = SessionStorage()
X_raw, y, trial_ids, session_indices, label_names, _ = storage.load_all_for_training()

extractor = EMGFeatureExtractor(channels=HAND_CHANNELS, cross_channel=True)
X = extractor.extract_features_batch(X_raw).astype(np.float64)

# Per-session normalization (same as EMGClassifier._apply_session_normalization)
from sklearn.preprocessing import StandardScaler
for sid in np.unique(session_indices):
    mask = session_indices == sid
    sc = StandardScaler()
    X[mask] = sc.fit_transform(X[mask])

feat_names = extractor.get_feature_names(n_channels=len(HAND_CHANNELS))
n_cls = len(np.unique(y))

# ─── Feature subset indices ───────────────────────────────────────────────────
TD_FEAT = ['rms','wl','zc','ssc','mav','var','iemg','wamp','ar1','ar2','ar3','ar4']
FD_FEAT = ['mnf','mdf','pkf','mnp','bp0','bp1','bp2','bp3']

td_idx = [i for i,n in enumerate(feat_names) if any(n.endswith(f'_{f}') for f in TD_FEAT)]
fd_idx = [i for i,n in enumerate(feat_names) if any(n.endswith(f'_{f}') for f in FD_FEAT)]
cc_idx = [i for i,n in enumerate(feat_names) if n.startswith('cov_') or n.startswith('cor_')]

print(f"Feature subsets — TD: {len(td_idx)}, FD: {len(fd_idx)}, CC: {len(cc_idx)}")

X_td = X[:, td_idx]
X_fd = X[:, fd_idx]
X_cc = X[:, cc_idx]

# ─── Train specialist LDAs with out-of-fold stacking ─────────────────────────
gkf = GroupKFold(n_splits=5)

print("Training specialist LDAs (out-of-fold for stacking)...")
lda_td = LinearDiscriminantAnalysis()
lda_fd = LinearDiscriminantAnalysis()
lda_cc = LinearDiscriminantAnalysis()

oof_td = cross_val_predict(lda_td, X_td, y, cv=gkf, groups=trial_ids, method='predict_proba')
oof_fd = cross_val_predict(lda_fd, X_fd, y, cv=gkf, groups=trial_ids, method='predict_proba')
oof_cc = cross_val_predict(lda_cc, X_cc, y, cv=gkf, groups=trial_ids, method='predict_proba')

# Specialist CV accuracy (for diagnostics)
for name, mdl, Xs in [('LDA_TD', lda_td, X_td), ('LDA_FD', lda_fd, X_fd), ('LDA_CC', lda_cc, X_cc)]:
    sc = cross_val_score(mdl, Xs, y, cv=gkf, groups=trial_ids)
    print(f"  {name}: {sc.mean()*100:.1f}% ± {sc.std()*100:.1f}%")

# ─── Train meta-LDA on out-of-fold outputs ───────────────────────────────────
X_meta = np.hstack([oof_td, oof_fd, oof_cc])   # (n_samples, 3*n_cls = 15)
meta_lda = LinearDiscriminantAnalysis()
meta_sc = cross_val_score(meta_lda, X_meta, y, cv=gkf, groups=trial_ids)
print(f"  Meta-LDA: {meta_sc.mean()*100:.1f}% ± {meta_sc.std()*100:.1f}%")

# Fit all models on full dataset for deployment
lda_td.fit(X_td, y); lda_fd.fit(X_fd, y); lda_cc.fit(X_cc, y)
meta_lda.fit(X_meta, y)

# ─── Export all weights to C header ──────────────────────────────────────────
def lda_to_c_arrays(lda, name, feat_dim, n_cls, label_names, class_order):
    """Generate C array strings for LDA weights and intercepts."""
    # Reorder classes to match label_names order
    coef = lda.coef_    # shape (n_cls, feat_dim) for LinearDiscriminantAnalysis
    intercept = lda.intercept_
    lines = []
    lines.append(f"const float {name}_WEIGHTS[{n_cls}][{feat_dim}] = {{")
    for c in class_order:
        row = ', '.join(f'{v:.8f}f' for v in coef[c])
        lines.append(f"    {{{row}}},  // {label_names[c]}")
    lines.append("};")
    lines.append(f"const float {name}_INTERCEPTS[{n_cls}] = {{")
    intercept_str = ', '.join(f'{intercept[c]:.8f}f' for c in class_order)
    lines.append(f"    {intercept_str}")
    lines.append("};")
    return '\n'.join(lines)

class_order = list(range(n_cls))
out_path = Path('EMG_Arm/src/core/model_weights_ensemble.h')

with open(out_path, 'w') as f:
    f.write("// Auto-generated by train_ensemble.py — do not edit\n")
    f.write("#pragma once\n\n")
    f.write(f"#define MODEL_NUM_CLASSES    {n_cls}\n")
    f.write(f"#define MODEL_NUM_FEATURES   {X.shape[1]}\n")
    f.write(f"#define ENSEMBLE_PER_CH_FEATURES 20\n\n")
    f.write(f"#define TD_FEAT_OFFSET  {min(td_idx)}\n")
    f.write(f"#define TD_NUM_FEATURES {len(td_idx)}\n")
    f.write(f"#define FD_FEAT_OFFSET  {min(fd_idx)}\n")
    f.write(f"#define FD_NUM_FEATURES {len(fd_idx)}\n")
    f.write(f"#define CC_FEAT_OFFSET  {min(cc_idx)}\n")
    f.write(f"#define CC_NUM_FEATURES {len(cc_idx)}\n")
    f.write(f"#define META_NUM_INPUTS ({3} * MODEL_NUM_CLASSES)\n\n")

    f.write(lda_to_c_arrays(lda_td,  'LDA_TD',  len(td_idx), n_cls, label_names, class_order))
    f.write('\n\n')
    f.write(lda_to_c_arrays(lda_fd,  'LDA_FD',  len(fd_idx), n_cls, label_names, class_order))
    f.write('\n\n')
    f.write(lda_to_c_arrays(lda_cc,  'LDA_CC',  len(cc_idx), n_cls, label_names, class_order))
    f.write('\n\n')
    f.write(lda_to_c_arrays(meta_lda, 'META_LDA', 3*n_cls,   n_cls, label_names, class_order))
    f.write('\n\n')

    names_str = ', '.join(f'"{label_names[c]}"' for c in class_order)
    f.write(f"const char *MODEL_CLASS_NAMES[MODEL_NUM_CLASSES] = {{{names_str}}};\n")

print(f"Exported ensemble weights to {out_path}")
print(f"Total weight storage: {(len(td_idx)+len(fd_idx)+len(cc_idx)+3*n_cls)*n_cls*4} bytes float32")
```

**Note on LinearDiscriminantAnalysis with multi-class**: scikit-learn's LDA uses a
`(n_classes-1, n_features)` coef matrix for multi-class. Verify `lda.coef_.shape` after
fitting — if it is `(n_cls-1, n_feat)` rather than `(n_cls, n_feat)`, use the
`decision_function()` output structure and adjust the export accordingly.

---

# PART VII — FEATURE SELECTION FOR ESP32 PORTING

After Change 1 is trained, use this to decide what to port to C firmware.

### Step 1 — Get Feature Importance

```python
importance = np.abs(classifier.model.coef_).mean(axis=0)
feat_names  = classifier.feature_extractor.get_feature_names(n_channels=len(HAND_CHANNELS))
ranked = sorted(zip(feat_names, importance), key=lambda x: -x[1])
print("Top 20 features by LDA discriminative weight:")
for name, score in ranked[:20]:
    print(f"  {name:<35} {score:.4f}")
```

### Step 2 — Port Decision Matrix

| Feature | C Complexity | Prereq | Port? |
|---------|-------------|--------|-------|
| RMS, WL, ZC, SSC | ✓ Already in C | — | Keep |
| MAV, VAR, IEMG | Very easy (1 loop) | None | ✓ Yes |
| WAMP | Very easy (threshold on diff) | None | ✓ Yes |
| Cross-ch covariance | Easy (3×3 outer product) | None | ✓ Yes |
| Cross-ch correlation | Easy (normalize covariance) | Covariance | ✓ Yes |
| Bandpower bp0–bp3 | Medium (128-pt FFT via esp-dsp) | Add FFT call | ✓ Yes — highest ROI |
| MNF, MDF, PKF, MNP | Easy after FFT | Bandpower FFT | ✓ Free once FFT added |
| AR(4) | Medium (Levinson-Durbin in C) | None | Only if top-8 importance |

Once `dsps_fft2r_fc32()` is added for bandpower, MNF/MDF/PKF/MNP come free.

### Step 3 — Adding FFT-Based Features to inference.c

Add inside `compute_features()` loop, after time-domain features per channel:

```c
// 128-pt FFT for frequency-domain features per channel
// Zero-pad signal from INFERENCE_WINDOW_SIZE (150) to 128 by truncating
float fft_buf[256] = {0};  // 128 complex floats
for (int i = 0; i < 128 && i < INFERENCE_WINDOW_SIZE; i++) {
    fft_buf[2*i]   = signal[i];  // real
    fft_buf[2*i+1] = 0.0f;       // imag
}
dsps_fft2r_fc32(fft_buf, 128);
dsps_bit_rev_fc32(fft_buf, 128);

// Bandpower: bin k → freq = k * 1000/128 ≈ k * 7.8125 Hz
// Band 0: 20–80 Hz  → bins  3–10
// Band 1: 80–150 Hz → bins 10–19
// Band 2: 150–300 Hz→ bins 19–38
// Band 3: 300–500 Hz→ bins 38–64
int band_bins[5] = {3, 10, 19, 38, 64};
float bp[4] = {0,0,0,0};
for (int b = 0; b < 4; b++)
    for (int k = band_bins[b]; k < band_bins[b+1]; k++) {
        float re = fft_buf[2*k], im = fft_buf[2*k+1];
        bp[b] += re*re + im*im;
    }
// Store at correct indices (base = ch * 20)
int base = ch * 20;
features_out[base+16] = bp[0]; features_out[base+17] = bp[1];
features_out[base+18] = bp[2]; features_out[base+19] = bp[3];
```

---

# PART VIII — MEASUREMENT AND VALIDATION

## Baseline Protocol

**Run this BEFORE any change and after EACH change.**

```
1. python learning_data_collection.py → option 3 (Train Classifier)
2. Record:
   - "Mean CV accuracy: XX.X% ± Y.Y%"  (cross-validation)
   - Confusion matrix (which gesture pairs are most confused)
   - Per-gesture accuracy breakdown
3. On-device test:
   - Put on sensors, perform 10 reps of each gesture
   - Log classification output (UART or Python serial monitor)
   - Compute per-gesture accuracy manually
4. Record REST false-trigger rate: hold arm at rest for 30 seconds,
   count number of non-REST outputs
```

## Results Log

| Change | CV Acc Before | CV Acc After | Delta | On-Device Acc | False Triggers/30s | Keep? |
|--------|--------------|-------------|-------|---------------|-------------------|-------|
| Baseline | — | — | — | — | — | — |
| Change C (reject) | — | — | — | — | — | — |
| Change B (filter) | — | — | — | — | — | — |
| Change 0 (label shift) | — | — | — | — | — | — |
| Change 1 (features) | — | — | — | — | — | — |
| Change D (NVS calib) | — | — | — | — | — | — |
| Change 3 (augment) | — | — | — | — | — | — |
| Change 5 (benchmark) | — | — | — | — | — | — |
| Change 7+F (ensemble) | — | — | — | — | — | — |
| Change E (MLP) | — | — | — | — | — | — |

## When to Add More Gestures

| CV Accuracy | Recommendation |
|-------------|----------------|
| <80% | Do NOT add gestures — fix the existing 5 first |
| 80–90% | Adding 1–2 gestures is reasonable; expect 5–8% drop per new gesture |
| >90% | Good baseline; can add gestures; target staying above 85% |
| >95% | Excellent; can be ambitious with gesture count |

---

# PART IX — EXPORT WORKFLOW

## Path 1 — LDA / Ensemble (Changes 0–4, 7+F)

```
1. Train: python learning_data_collection.py → option 3  (single LDA)
         OR: python train_ensemble.py                     (full ensemble)

2. Export:
   Single LDA:  classifier.export_to_header(Path('EMG_Arm/src/core/model_weights.h'))
   Ensemble:    export_ensemble_header() in train_ensemble.py
                → writes model_weights_ensemble.h

3. Port new features to inference.c (if Change 1 features added):
   - Follow feature selection decision matrix (Part VII)
   - CRITICAL: C feature index order MUST match Python FEATURE_ORDER exactly

4. Build + flash: pio run -t upload
```

## Path 2 — int8 MLP via TFLM (Change E)

```
1. python train_mlp_tflite.py  → emg_model_data.cc
2. Add TFLM to platformio.ini lib_deps
3. Replace LDA inference call with inference_mlp_predict() in inference.c
   OR use inference_ensemble_predict() which calls MLP as fallback (Change F)
4. pio run -t upload
```

## Feature Index Contract (Critical)

The order of values written to `features_out[]` in `compute_features()` in C **must exactly
match** `FEATURE_ORDER` in `extract_features_window()` in Python, index for index.

To verify before flashing: print both the C feature names (from `MODEL_FEATURE_NAMES` if
added to header) and Python `extractor.get_feature_names()` and diff them.

---

# PART X — REFERENCES

**Primary paper**: Kaifosh, P., Reardon, T., et al. "A high-bandwidth neuromotor prosthesis
enabled by implicit information in intrinsic motor neurons." *Nature* (2025).
doi:10.1038/s41586-025-09255-w

**Meta codebase** (label alignment, CLER metric, model architectures):
`C:/VSCode/Marvel_Projects/Meta_Emg_Stuff/generic-neuromotor-interface/`
- `data.py`: onset detection, `searchsorted` alignment, window jitter
- `cler.py`: threshold=0.35, debounce=50ms, tolerance=±50/250ms
- `networks.py`: model architectures, left_context=20, stride=10
- `lightning.py`: `targets[..., left_context::stride]` label shift

**Barachant et al. 2012**: "Multiclass brain–computer interface classification by
Riemannian geometry." — matrix logarithm reference (MPF features).

**Espressif libraries**:
- esp-dsp: `github.com/espressif/esp-dsp` — biquad, FFT, dot-product
- esp-dl: `github.com/espressif/esp-dl` — quantized MLP/CNN inference
- TFLite Micro: `github.com/tensorflow/tflite-micro`

**All project files** (existing + planned):

```
── Laptop / Python ─────────────────────────────────────────────────────────────────────────
C:/VSCode/Marvel_Projects/Bucky_Arm/learning_data_collection.py  ← main: data collection + training
C:/VSCode/Marvel_Projects/Bucky_Arm/live_predict.py              ← NEW (Part 0.6): laptop-side live inference
C:/VSCode/Marvel_Projects/Bucky_Arm/train_ensemble.py            ← NEW (Change 7): ensemble training
C:/VSCode/Marvel_Projects/Bucky_Arm/train_mlp_tflite.py          ← NEW (Change E): int8 MLP export

── ESP32 Firmware — Existing ───────────────────────────────────────────────────────────────
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/platformio.ini
  └─ ADD lib_deps: espressif/esp-dsp (Changes B,1,F), tensorflow/tflite-micro (Change E)
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/config/config.h
  └─ MODIFY: remove system_mode_t; add EMG_STANDALONE to MAIN_MODE enum (Part 0.7, S1)
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/app/main.c
  └─ MODIFY: add STATE_LAPTOP_PREDICT, CMD_START_LAPTOP_PREDICT, run_laptop_predict_loop(),
             run_standalone_loop() (Part 0.5)
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/drivers/emg_sensor.c
  └─ MODIFY (Change A): migrate from adc_oneshot to adc_continuous driver
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/inference.c
  └─ MODIFY: add inference_get_gesture_by_name(), IIR filter (B), features (1), confidence rejection (C)
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/inference.h
  └─ MODIFY: add inference_get_gesture_by_name() declaration
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/gestures.c
  └─ MODIFY: update gesture_names[] and gestures_execute() when adding gestures
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/model_weights.h
  └─ AUTO-GENERATED by export_to_header() — do not edit manually

── ESP32 Firmware — New Files ──────────────────────────────────────────────────────────────
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/bicep.h/.c        ← Part 0 / Section 2.2
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/calibration.h/.c  ← Change D (NVS z-score)
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/inference_ensemble.h/.c  ← Change F
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/inference_mlp.h/.cc      ← Change E
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/model_weights_ensemble.h ← AUTO-GENERATED (Change 7)
C:/VSCode/Marvel_Projects/Bucky_Arm/EMG_Arm/src/core/emg_model_data.h/.cc     ← AUTO-GENERATED (Change E)
```
