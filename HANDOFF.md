# Bucky Arm — Inference Consistency Handoff

**Last updated:** 2026-06-11
**Branch:** `main` · last commit before this work: `9b38037 updated train/serve and python/firmware consistency`
**Status:** Source code is fully consistent across Python / firmware / train / serve. **The deployed firmware artifacts are stale and must be regenerated before testing on-device.**

---

## 1. What this project is

EMG-controlled prosthetic hand. An ESP32-S3 reads 4 EMG channels at 1 kHz, classifies
hand gestures from forearm channels 0–2 (channel 3 = bicep, handled separately), and
drives servos via a PCA9685.

Three classifiers share **one** 69-dimensional feature vector and are averaged:
- **Single LDA** (exported to `model_weights.h`)
- **3-specialist + meta-LDA ensemble** (`model_weights_ensemble.h`)
- **int8 MLP** via TFLite Micro (`emg_model_data.cc`)

Gestures (alphabetical class order, used **everywhere**): `fist, hook_em, open, rest, thumbs_up`.

### Key constants (identical Python ↔ firmware)
| Constant | Value | Python | Firmware |
|---|---|---|---|
| Sample rate | 1000 Hz | `SAMPLING_RATE_HZ` | `EMG_SAMPLE_RATE_HZ` |
| Window | 150 samples (150 ms) | `WINDOW_SIZE_MS` | `INFERENCE_WINDOW_SIZE` |
| Hop/stride | 25 samples | `HOP_SIZE_MS` | `INFERENCE_HOP_SIZE` |
| FFT size | 256 | `fft_n` | `FFT_N` |
| Hand channels | [0,1,2] (3) | `HAND_CHANNELS` | `HAND_NUM_CHANNELS` |
| Total channels | 4 | `NUM_CHANNELS` | `EMG_NUM_CHANNELS` |
| Reinhard map | `64·x/(32+|x|)` | — | — |
| ZC/SSC threshold | 0.1 × rms | `zc/ssc_threshold_percent` | `FEAT_ZC/SSC_THRESH` |

---

## 2. The 69-feature layout (must match exactly)

Per channel (20), in order:
`rms, wl, zc, ssc, mav, var, iemg, wamp, ar1..ar4, mnf, mdf, pkf, mnp, bp0..bp3`
- 3 channels × 20 = 60
- Cross-channel (9): pairs (0,1),(0,2),(1,2) × (corr, lrms, cov) → indices 60–68

Ensemble feature subsets (gathered from the 69-vector):
- **TD** (36): indices {0–11, 20–31, 40–51}
- **FD** (24): indices {12–19, 32–39, 52–59}
- **CC** (9): indices {60–68}
- Meta-LDA input order: `[td_probs, fd_probs, cc_probs]`

---

## 3. The pipeline, end to end

### Feature extraction (identical in train / live / firmware)
1. Causal IIR bandpass 20–450 Hz (2nd-order Butterworth, 2 cascaded biquads DF2T).
   Coefficients verified bit-exact to `scipy.signal.butter(2,[20,450],fs=1000,'band',sos)`.
2. Per-window DC removal (mean subtract).
3. Reinhard tone-map.
4. Compute features. `var` = true variance (`mean(x²) − mean(x)²`), spectral uses
   129 bins (`rfft`, incl. Nyquist).

### Normalization / calibration
- **Train:** per-session z-score. `mu` = class-balanced mean; `sigma` = per-session
  std + 1e-8. `sigma_train` = mean per-session sigma, computed on **real (non-augmented)
  rows only**, baked into firmware as `MODEL_FEAT_STD`.
- **Serve (firmware + live_predict):** `(x − rest_mean) / sigma_train`.
- **Serve (GUI):** `CalibrationTransform.apply` = `(x − mu_calib) / sigma_train`.

### Model combination (identical: firmware / live_predict / GUI)
`avg( LDA softmax, ensemble meta softmax, MLP soft-one-hot )`
- MLP "soft one-hot" = winner gets its confidence, remaining mass split evenly across
  the other K−1 classes.

### Post-processing (identical everywhere)
EMA 0.70 → reject if smoothed peak < 0.40 (hold last output) → majority vote (window 5)
→ debounce (3 consecutive to switch).

---

## 4. The single source of truth (don't duplicate these)

| Concern | Location |
|---|---|
| Feature switches (reinhard / expanded / normalize) | `FEATURE_REINHARD/EXPANDED/NORMALIZE` + `make_feature_extractor()` in `learning_data_collection.py` |
| Train pipeline (augment → extract → normalize) | `build_training_matrix()` in `learning_data_collection.py` |
| Per-session z-score (real-only stats) | `_session_zscore()` in `learning_data_collection.py` |
| Post-processing smoother | `PredictionSmoother` (firmware mirror: `vote_postprocess()` in `main.c`) |

All three training scripts (`EMGClassifier.train`, `train_ensemble.py`, `train_mlp_tflite.py`)
call `build_training_matrix()` → **byte-identical inputs** (same augmentation seed=42).

---

## 5. What changed (this work)

### Firmware (`EMG_Arm/src/core/`)
- `emg_sensor.c` — removed per-sample debug printf that flooded UART and broke real-time.
- `inference.c` — `var` now true variance (− mean²); FFT now 129 bins (`k <= FFT_N/2`,
  `mnp /= FFT_N/2+1`) to match numpy `rfft`.
- `calibration.c` — `calibration_set_train_scale()` overrides REST std with baked
  `MODEL_FEAT_STD` (sigma_train) when present; REST mean kept as offset.
- `app/main.c` — removed wrong `gestures_index_to_gesture` mapping; multi-model voting
  via `vote_postprocess` (EMA 0.7 / reject 0.4 / vote 5 / debounce 3).

### Python — feature/train unification (`learning_data_collection.py`)
- Added `FEATURE_*` ground-truth flags + `make_feature_extractor()`.
- Added `build_training_matrix()` + `_session_zscore()` (real-only normalization stats).
- `train()` rewired to `build_training_matrix`; dropped dead MPF branch.
- `extract_features_batch` uses **causal** `sosfilt` (was zero-phase `sosfiltfilt`).
- Cross-channel filter order fixed (bandpass → DC → reinhard, matches per-channel).
- `export_to_header` emits `MODEL_FEAT_STD` + `MODEL_HAS_TRAIN_STD`.
- `PredictionSmoother` gained optional `reject_threshold` (default 0.0 = off).

### Python — training scripts
- `train_ensemble.py` / `train_mlp_tflite.py` — use `build_training_matrix` (augment 3×,
  real-only stats). Ensemble: `trial_ids` tiled so `GroupKFold` OOF stacking can't leak.
  MLP: shuffle before `fit` (augment order otherwise made `validation_split` tail pure-aug).

### Python — serve paths
- `live_predict.py` — calibration scale now `sigma_train`; MLP soft-one-hot; uses
  `PredictionSmoother(0.7/5/3, reject=0.40)`.
- `emg_gui.py` — laptop loop: removed energy-gate short-circuit; MLP soft-one-hot;
  smoother `debounce 4→3` + `reject_threshold=0.40`.

---

## 6. Accepted residuals (DO NOT "fix" — being tested empirically)

1. **Filter warmup** — firmware filters continuously across windows; train/live filter
   each window from `zi = x0`. Differs only in the first ~10–30 of 150 samples.
2. **Calibration mean** — serve subtracts REST-only mean; train uses class-balanced mean
   (GUI uses its `CalibrationTransform` mean). The *scale* (`sigma_train`) is consistent
   everywhere; only the mean differs.

If accuracy is poor and these are suspected, options: (a) multi-gesture on-device
calibration to match the class-balanced training mean, or (b) bake a fixed `mu_train`.

---

## 7. BEFORE TESTING — regenerate artifacts (required)

The committed headers predate this work (e.g. `model_weights.h` has no `MODEL_FEAT_STD`).
Retrain + re-export all three so the binaries match the consistent code:

```
# 1. Single LDA + sigma_train  → model_weights.h   (via the GUI "Train" + "Export header")
python emg_gui.py        # Train (uses load_all_for_training + session_indices), then Export

# 2. Ensemble → model_weights_ensemble.h
python train_ensemble.py

# 3. MLP → emg_model_data.cc  (needs tensorflow)
python train_mlp_tflite.py
```

Then verify `EMG_Arm/src/core/model_weights.h` contains:
- `#define MODEL_HAS_TRAIN_STD 1` and a `MODEL_FEAT_STD[69]` array
- `MODEL_USE_REINHARD 1`, `MODEL_EXPAND_FEATURES 1`, `MODEL_NORMALIZE_FEATURES 1`
- To enable ensemble/MLP on-device: set `MODEL_USE_ENSEMBLE 1` / `MODEL_USE_MLP 1`

Build/flash:
```
cd EMG_Arm && pio run -t upload
```

---

## 8. How to test

### A. Replay test (no hardware EMG needed) — fastest sanity check
Set `LIVE_EMG=0` to stream recorded data through `emg_sensor_read` (the original bug-repro
path). Confirms the on-device pipeline classifies known data correctly.

### B. Laptop inference vs on-device (apples-to-apples)
1. Set `MAIN_MODE = EMG_MAIN` in `config.h`, flash.
2. `python live_predict.py --port COMx`
   - Holds REST 3 s for calibration, then classifies; sends `{"gesture":...}` back.
   - Now uses sigma_train scale + firmware-matched averaging/smoothing → should track the
     on-device output closely (differences = the two accepted residuals + int8-vs-float MLP).

### C. On-device standalone
`MAIN_MODE = REAL_MAIN`, flash, perform gestures. `run_standalone_loop` prints per-class
averaged probabilities (`name:prob`) each hop for debugging.

### What "working" looks like
- REST is stable (no spurious actuation) — reject gate (0.40) + debounce should hold it.
- Each trained gesture is reachable and reasonably stable once held ~3 hops (debounce).
- If predictions collapse to one class → suspect calibration scale (check `MODEL_FEAT_STD`
  is present and non-trivial) before anything else.

---

## 9. Debugging map (where to look)

| Symptom | First suspect | File |
|---|---|---|
| All predictions one class / saturated softmax | calibration scale (sigma_train missing) | `calibration.c`, `model_weights.h` |
| Laptop ≠ on-device beyond small margin | feature parity / calibration mean | `inference.c` vs `learning_data_collection.py` |
| Ensemble disagrees with single LDA | gather offsets / class order | `inference_ensemble.c`, `train_ensemble.py` |
| Jittery output | smoother params | `main.c` `vote_postprocess`, `PredictionSmoother` |
| Garbage features | filter coeffs / window fill | `inference.c` IIR + `compute_features_expanded` |

`inference_predict()` in `inference.c` is **dead code** (active path is
`inference_predict_raw` + `vote_postprocess`); ignore it (candidate for deletion).

---

## 10. Memory / further context

Persistent notes live in
`C:\Users\surya\.claude\projects\C--Software-Dev-Marvel-Projects-Bucky-Arm\memory\`:
- `project_inference_bugs.md` — full bug-hunt + consistency audit log (most detailed).
- `project_emg_replay.md` — flash-backed replay design/status.
