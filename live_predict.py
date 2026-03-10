"""
live_predict.py — Laptop-side live EMG inference for Bucky Arm.

Use this script when the ESP32 is in EMG_MAIN mode and you want the laptop to
run the classifier (instead of the on-device model). Useful for:
  - Comparing laptop accuracy vs. on-device accuracy before flashing a new model
  - Debugging the feature pipeline without reflashing firmware
  - Running an updated model that hasn't been exported to C yet

Workflow:
  1. ESP32 must be in EMG_MAIN mode (MAIN_MODE = EMG_MAIN in config.h)
  2. This script handshakes → requests STATE_LAPTOP_PREDICT
  3. ESP32 streams raw ADC CSV at 1 kHz
  4. Script collects 3s of REST for session normalization, then classifies
  5. Every 25ms (one hop), the predicted gesture is sent back: {"gesture":"fist"}
  6. ESP32 executes the received gesture command on the arm

Usage:
    python live_predict.py --port COM3
    python live_predict.py --port COM3 --model models/my_model.joblib
    python live_predict.py --port COM3 --confidence 0.45
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import serial

# Import from the main training pipeline
sys.path.insert(0, str(Path(__file__).parent))
from learning_data_collection import (
    EMGClassifier,
    NUM_CHANNELS,
    SAMPLING_RATE_HZ,
    WINDOW_SIZE_MS,
    HOP_SIZE_MS,
    HAND_CHANNELS,
)

# Derived constants
WINDOW_SIZE = int(WINDOW_SIZE_MS * SAMPLING_RATE_HZ / 1000)  # 150 samples
HOP_SIZE    = int(HOP_SIZE_MS    * SAMPLING_RATE_HZ / 1000)  # 25 samples
BAUD_RATE   = 921600
CALIB_SECS  = 3.0   # seconds of REST to collect at startup for normalization

# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port",       required=True,
                   help="Serial port (e.g. COM3 on Windows, /dev/ttyUSB0 on Linux)")
    p.add_argument("--model",      default=None,
                   help="Path to .joblib model file. Defaults to the most recently trained model.")
    p.add_argument("--confidence", type=float, default=0.40,
                   help="Reject predictions below this confidence (default: 0.40, same as firmware)")
    p.add_argument("--no-calib",   action="store_true",
                   help="Skip REST calibration (use raw features — only for quick testing)")
    return p.parse_args()


def load_model(model_path: str | None) -> EMGClassifier:
    if model_path:
        path = Path(model_path)
    else:
        path = EMGClassifier.get_latest_model_path()
        if path is None:
            print("[ERROR] No trained model found in models/. Run training first.")
            sys.exit(1)
        print(f"[Model] Auto-selected latest model: {path.name}")

    classifier = EMGClassifier.load(path)
    if not classifier.is_trained:
        print("[ERROR] Loaded model is not trained.")
        sys.exit(1)
    return classifier


def handshake(ser: serial.Serial) -> bool:
    """Send connect command and wait for ack_connect from ESP32."""
    print("[Handshake] Sending connect command...")
    ser.write(b'{"cmd":"connect"}\n')
    deadline = time.time() + 5.0
    while time.time() < deadline:
        raw = ser.readline()
        line = raw.decode("utf-8", errors="ignore").strip()
        if "ack_connect" in line:
            print(f"[Handshake] Connected — {line}")
            return True
        if line:
            print(f"[Handshake] (ignored) {line}")
    print("[ERROR] No ack_connect within 5s. Is the ESP32 powered and in EMG_MAIN mode?")
    return False


def collect_calibration_windows(
    ser: serial.Serial,
    n_windows: int,
    extractor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect n_windows of REST EMG, extract features, and compute
    per-feature mean and std for session normalization.

    Returns (mean, std) arrays of shape (n_features,).
    """
    print(f"[Calib] Hold arm relaxed at rest for {CALIB_SECS:.0f}s...")

    raw_buf   = np.zeros((WINDOW_SIZE, NUM_CHANNELS), dtype=np.float32)
    samples   = 0
    feat_list = []

    while len(feat_list) < n_windows:
        raw = ser.readline()
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line or line.startswith("{"):
            continue
        try:
            vals = [float(v) for v in line.split(",")]
        except ValueError:
            continue
        if len(vals) != NUM_CHANNELS:
            continue

        # Slide window
        raw_buf = np.roll(raw_buf, -1, axis=0)
        raw_buf[-1] = vals
        samples += 1

        if samples >= WINDOW_SIZE and (samples % HOP_SIZE) == 0:
            feat = extractor.extract_features_window(raw_buf)
            feat_list.append(feat)

            done = len(feat_list)
            if done % 10 == 0:
                pct = int(100 * done / n_windows)
                print(f"  {pct}% ({done}/{n_windows} windows)", end="\r", flush=True)

    print(f"\n[Calib] Collected {len(feat_list)} windows.")
    feats = np.array(feat_list, dtype=np.float32)
    mean  = feats.mean(axis=0)
    std   = np.where(feats.std(axis=0) > 1e-6, feats.std(axis=0), 1e-6).astype(np.float32)
    print("[Calib] Session normalization computed.")
    return mean, std


def load_ensemble():
    """Load ensemble sklearn models if available."""
    path = Path(__file__).parent / 'models' / 'emg_ensemble.joblib'
    if not path.exists():
        return None
    try:
        import joblib
        ens = joblib.load(path)
        print(f"[Model] Loaded ensemble (4 LDAs)")
        return ens
    except Exception as e:
        print(f"[Model] Ensemble load failed: {e}")
        return None


def load_mlp():
    """Load MLP numpy weights if available."""
    path = Path(__file__).parent / 'models' / 'emg_mlp_weights.npz'
    if not path.exists():
        return None
    try:
        mlp = dict(np.load(path, allow_pickle=True))
        print(f"[Model] Loaded MLP weights (numpy)")
        return mlp
    except Exception as e:
        print(f"[Model] MLP load failed: {e}")
        return None


def run_ensemble(ens, features):
    """Run ensemble: 3 specialist LDAs → meta-LDA → probabilities."""
    p_td = ens['lda_td'].predict_proba([features[ens['td_idx']]])[0]
    p_fd = ens['lda_fd'].predict_proba([features[ens['fd_idx']]])[0]
    p_cc = ens['lda_cc'].predict_proba([features[ens['cc_idx']]])[0]
    x_meta = np.concatenate([p_td, p_fd, p_cc])
    return ens['meta_lda'].predict_proba([x_meta])[0]


def run_mlp(mlp, features):
    """Run MLP forward pass: Dense(32,relu) → Dense(16,relu) → Dense(5,softmax)."""
    x = features.astype(np.float32)
    x = np.maximum(0, x @ mlp['w0'] + mlp['b0'])
    x = np.maximum(0, x @ mlp['w1'] + mlp['b1'])
    logits = x @ mlp['w2'] + mlp['b2']
    e = np.exp(logits - logits.max())
    return e / e.sum()


def main():
    args = parse_args()

    # ── Load classifier ──────────────────────────────────────────────────────
    classifier = load_model(args.model)
    extractor  = classifier.feature_extractor
    ensemble   = load_ensemble()
    mlp        = load_mlp()
    model_names = ["LDA"]
    if ensemble:
        model_names.append("Ensemble")
    if mlp:
        model_names.append("MLP")
    print(f"[Model] Active: {' + '.join(model_names)} ({len(model_names)} models)")

    # ── Open serial ──────────────────────────────────────────────────────────
    try:
        ser = serial.Serial(args.port, BAUD_RATE, timeout=1.0)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open {args.port}: {e}")
        sys.exit(1)

    time.sleep(0.5)
    ser.reset_input_buffer()

    # ── Handshake ────────────────────────────────────────────────────────────
    if not handshake(ser):
        ser.close()
        sys.exit(1)

    # ── Request laptop-predict mode ──────────────────────────────────────────
    ser.write(b'{"cmd":"start_laptop_predict"}\n')
    print("[Control] ESP32 entering STATE_LAPTOP_PREDICT — streaming ADC...")

    # ── Calibration ──────────────────────────────────────────────────────────
    calib_mean = None
    calib_std  = None
    if not args.no_calib:
        n_calib = max(20, int(CALIB_SECS * 1000 / HOP_SIZE_MS))
        calib_mean, calib_std = collect_calibration_windows(ser, n_calib, extractor)
    else:
        print("[Calib] Skipped (--no-calib). Accuracy may be reduced.")

    # ── Live prediction loop ─────────────────────────────────────────────────
    print(f"\n[Predict] Running. Confidence threshold: {args.confidence:.2f}")
    print("[Predict] Press Ctrl+C to stop.\n")

    raw_buf      = np.zeros((WINDOW_SIZE, NUM_CHANNELS), dtype=np.float32)
    samples      = 0
    last_gesture = None
    n_inferences = 0
    n_rejected   = 0

    try:
        while True:
            raw  = ser.readline()
            line = raw.decode("utf-8", errors="ignore").strip()

            # Skip JSON telemetry lines from ESP32
            if not line or line.startswith("{"):
                continue

            # Parse CSV sample
            try:
                vals = [float(v) for v in line.split(",")]
            except ValueError:
                continue
            if len(vals) != NUM_CHANNELS:
                continue

            # Slide window
            raw_buf = np.roll(raw_buf, -1, axis=0)
            raw_buf[-1] = vals
            samples += 1

            # Classify every HOP_SIZE samples
            if samples >= WINDOW_SIZE and (samples % HOP_SIZE) == 0:
                feat = extractor.extract_features_window(raw_buf).astype(np.float32)

                # Apply session normalization
                if calib_mean is not None:
                    feat = (feat - calib_mean) / calib_std

                # Run all available models and average probabilities
                probas = [classifier.model.predict_proba([feat])[0]]
                if ensemble:
                    try:
                        probas.append(run_ensemble(ensemble, feat))
                    except Exception:
                        pass
                if mlp:
                    try:
                        probas.append(run_mlp(mlp, feat))
                    except Exception:
                        pass
                proba      = np.mean(probas, axis=0)
                class_idx  = int(np.argmax(proba))
                confidence = float(proba[class_idx])
                gesture    = classifier.label_names[class_idx]
                n_inferences += 1

                # Reject below threshold
                if confidence < args.confidence:
                    n_rejected += 1
                    continue

                # Send gesture command to ESP32
                cmd = f'{{"gesture":"{gesture}"}}\n'
                ser.write(cmd.encode("utf-8"))

                # Local logging (only on change)
                if gesture != last_gesture:
                    reject_rate = 100 * n_rejected / n_inferences if n_inferences else 0
                    print(f"  → {gesture:<12}  conf={confidence:.2f}  "
                          f"reject_rate={reject_rate:.0f}%")
                    last_gesture  = gesture
                    n_rejected    = 0
                    n_inferences  = 0

    except KeyboardInterrupt:
        print("\n\n[Stop] Sending stop command to ESP32...")
        ser.write(b'{"cmd":"stop"}\n')
        time.sleep(0.2)
        ser.close()
        print("[Stop] Done.")


if __name__ == "__main__":
    main()
