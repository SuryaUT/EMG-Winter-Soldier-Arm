"""
Compare the firmware's PARITY_DUMP against Python, feature by feature.

The firmware replays `replay_samples[]` (baked from a session HDF5) and prints,
per inference hop, its calibrated 69-feature vector plus the LDA / ensemble / MLP
outputs. This script feeds Python the SAME input and diffs the two.

Because the input is bit-identical on both sides, every difference reported here
is a pure implementation difference — no sensor noise, no timing, no confound.

Usage:
    python tools/parity_compare.py dump.log [--session updated010_20260214_204204.hdf5]
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from learning_data_collection import EMGFeatureExtractor, HAND_CHANNELS  # noqa: E402

WIN, HOP, FS = 150, 25, 1000.0


def err_pct(x):
    return f"{x*100:.3f}% sig"

ap = argparse.ArgumentParser()
ap.add_argument("dump", help="captured serial log from MAIN_MODE=PARITY_DUMP")
ap.add_argument("--session", default="updated010_20260214_204204.hdf5")
ap.add_argument("--data-dir", default="collected_data")
args = ap.parse_args()

# ---------------------------------------------------------------- parse dump
cmean = cstd = None
hops = []
for line in Path(args.dump).read_text(errors="replace").splitlines():
    line = line.strip()
    if line.startswith("CMEAN"):
        cmean = np.array([float(v) for v in line.split(",")[1:]])
    elif line.startswith("CSTD"):
        cstd = np.array([float(v) for v in line.split(",")[1:]])
    elif line.startswith("H,"):
        p = line.split(",")
        try:
            hops.append((int(p[1]), int(p[2]), np.array([float(v) for v in p[3:]])))
        except ValueError:
            print(f"[warn] skipping malformed hop line: {line[:60]}...")

if not hops or cmean is None or cstd is None:
    sys.exit("No parity records found. Did the dump complete (look for #PARITY_END)?")

NF, NC = 69, 5
expected = NF + NC + NC + 2
hops = [h for h in hops if len(h[2]) == expected]
print(f"Parsed {len(hops)} hops from {args.dump}")

fw_feat = np.stack([h[2][:NF] for h in hops])
fw_lda = np.stack([h[2][NF:NF + NC] for h in hops])
fw_ens = np.stack([h[2][NF + NC:NF + 2 * NC] for h in hops])
fw_mlp_cls = np.array([int(h[2][-2]) for h in hops])
fw_mlp_conf = np.array([h[2][-1] for h in hops])
end_idx = np.array([h[1] for h in hops])

# ------------------------------------------------------- rebuild Python side
# CRITICAL: the firmware sees uint16 ADC counts. hdf5_to_replay.py does
# round(clip(raw,0,65535)) -> uint16, so Python must round identically or every
# feature shows a spurious diff.
with h5py.File(Path(args.data_dir) / args.session, "r") as f:
    raw = np.clip(np.array(f["raw_samples/channels"]), 0, 65535).round().astype(np.uint16)
raw = raw.astype(np.float64)

# The firmware's biquad runs continuously from sample 0. inference_init() memsets
# its state to ZERO (not sosfilt_zi*x0), so seed with zeros to match exactly --
# otherwise the ~2300 mV DC offset makes the startup transient differ.
sos = butter(2, [20 / (FS / 2), 450 / (FS / 2)], btype="band", output="sos")
cont = np.empty_like(raw)
zi0 = np.zeros((sos.shape[0], 2))
for ch in range(raw.shape[1]):
    cont[:, ch], _ = sosfilt(sos, raw[:, ch], zi=zi0)

# bandpass=False: the stream is already filtered, matching the firmware's
# filter-then-window order.
ext = EMGFeatureExtractor(channels=HAND_CHANNELS, reinhard=True, expanded=True,
                          cross_channel=True, normalize=True, bandpass=False)

py_raw = np.stack([ext.extract_features_window(cont[i - WIN + 1:i + 1]) for i in end_idx])

# Compare in RAW feature space. Calibrated features are z-scored (~zero mean), so
# normalising by their mean magnitude blows up any feature centred near zero and
# reports meaningless 100%+ errors. Invert calibration_apply() instead:
#   raw = calibrated * std + mean
fw_raw = fw_feat * cstd + cmean

names = [f"ch{c}_{k}" for c in range(3) for k in EMGFeatureExtractor._EXPANDED_KEYS]
names += [f"cc{p}_{m}" for p in ("01", "02", "12") for m in ("corr", "lrms", "cov")]

# Scale each feature by its own spread across windows -- the only scale the model
# actually cares about -- so features with a large constant offset aren't flattered.
spread = py_raw.std(0) + 1e-12
err = np.abs(fw_raw - py_raw).mean(0) / spread

print("\n=== RAW FEATURES: firmware vs Python (error as fraction of each feature's sigma) ===")
print(f"  median : {err_pct(np.median(err))}")
print(f"  max    : {err_pct(np.max(err))}  ({names[int(np.argmax(err))]})")
print("\n  worst 12 features:")
for i in np.argsort(-err)[:12]:
    print(f"    {names[i]:14s} {err_pct(err[i]):>10s}   fw={fw_raw[:, i].mean():+.6g}  py={py_raw[:, i].mean():+.6g}")

TOL = 0.01  # 1% of a feature's sigma; pure float32 round-off sits far below this
bad = [i for i in range(NF) if err[i] > TOL]
if bad:
    print(f"\n  *** {len(bad)}/69 feature(s) exceed 1% of sigma — REAL divergence:")
    for i in sorted(bad, key=lambda j: -err[j]):
        print(f"      {names[i]:14s} {err_pct(err[i])}")
else:
    print("\n  All 69 features agree to <1% of sigma — feature extraction is faithful.")

# ------------------------------------------------------------------ model diff
print("\n=== MODEL OUTPUTS (fed the FIRMWARE's own feature vector, so this")
print("    isolates the model math from any feature drift) ===")
h = Path(__file__).resolve().parent.parent / "EMG_Arm/src/core/model_weights.h"
import re
txt = h.read_text()


def arr(name):
    m = re.search(name + r"\s*\[[^\]]*\](?:\s*\[[^\]]*\])?\s*=\s*\{(.*?)\};", txt, re.S)
    return np.array([float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", m.group(1).replace("f", ""))])


W = arr("LDA_WEIGHTS").reshape(NC, NF)
B = arr("LDA_INTERCEPTS")
s = fw_feat @ W.T + B
py_lda = np.exp(s - s.max(1, keepdims=True))
py_lda /= py_lda.sum(1, keepdims=True)

lda_err = np.abs(py_lda - fw_lda).max()
print(f"  LDA softmax  max abs diff : {lda_err:.3e}  "
      f"({'OK' if lda_err < 1e-4 else 'DIVERGENT'})")
print(f"  LDA argmax   agreement    : {(py_lda.argmax(1) == fw_lda.argmax(1)).mean()*100:.2f}%")
print(f"  ensemble/MLP: firmware-only (needs ensemble header + TFLite to mirror);")
print(f"    ens argmax distribution : {np.bincount(fw_ens.argmax(1), minlength=NC)}")
print(f"    mlp class distribution  : {np.bincount(fw_mlp_cls[fw_mlp_cls >= 0], minlength=NC)}")
print(f"    mlp mean confidence     : {fw_mlp_conf.mean():.3f}")

print("\n=== CALIBRATION VECTORS (from the device) ===")
print(f"  mean : min={cmean.min():+.4g} max={cmean.max():+.4g}")
print(f"  std  : min={cstd.min():+.4g} max={cstd.max():+.4g}")
print("  (std should equal MODEL_FEAT_STD — the sigma_train override in calibration.c)")
fs_hdr = arr("MODEL_FEAT_STD")
d = np.abs(cstd - fs_hdr) / (np.abs(fs_hdr) + 1e-9)
print(f"  device std vs MODEL_FEAT_STD: max rel diff {d.max():.3e}  "
      f"({'OK' if d.max() < 1e-5 else 'MISMATCH'})")
