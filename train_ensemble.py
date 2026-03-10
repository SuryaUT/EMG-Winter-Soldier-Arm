"""
Train the full 3-specialist-LDA + meta-LDA ensemble.
Requires Change 1 (expanded features) to be implemented first.
Exports model_weights_ensemble.h for firmware Change F.

Architecture:
  LDA_TD (36 time-domain feat) ─┐
  LDA_FD (24 freq-domain feat)  ├─ 15 probs ─► Meta-LDA ─► final class
  LDA_CC (9  cross-ch feat)     ─┘

Change 7 — priority Tier 3.
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

# --- Load and extract features -----------------------------------------------
storage = SessionStorage()
X_raw, y, trial_ids, session_indices, label_names, _ = storage.load_all_for_training()

extractor = EMGFeatureExtractor(channels=HAND_CHANNELS, cross_channel=True, expanded=True, reinhard=True)
X = extractor.extract_features_batch(X_raw).astype(np.float64)

# Per-session class-balanced normalization
# Must match EMGClassifier._apply_session_normalization():
#   mean = average of per-class means (not overall mean), std = overall std.
# StandardScaler uses overall mean, which biases toward the majority class.
for sid in np.unique(session_indices):
    mask = session_indices == sid
    X_sess = X[mask]
    y_sess = y[mask]

    # Class-balanced mean: average of per-class centroids
    class_means = []
    for cls in np.unique(y_sess):
        class_means.append(X_sess[y_sess == cls].mean(axis=0))
    balanced_mean = np.mean(class_means, axis=0)

    # Overall std (same as StandardScaler)
    std = X_sess.std(axis=0)
    std[std < 1e-12] = 1.0  # avoid division by zero

    X[mask] = (X_sess - balanced_mean) / std

feat_names = extractor.get_feature_names(n_channels=len(HAND_CHANNELS))
n_cls = len(np.unique(y))

# --- Feature subset indices ---------------------------------------------------
# Per-channel layout (20 features/channel): indices 0-11 TD, 12-19 FD
# Cross-channel features start at index 60 (3 channels × 20 features each)
TD_FEAT = ['rms', 'wl', 'zc', 'ssc', 'mav', 'var', 'iemg', 'wamp', 'ar1', 'ar2', 'ar3', 'ar4']
FD_FEAT = ['mnf', 'mdf', 'pkf', 'mnp', 'bp0', 'bp1', 'bp2', 'bp3']

td_idx = [i for i, n in enumerate(feat_names)
          if any(n.endswith(f'_{f}') for f in TD_FEAT) and n.startswith('ch')]
fd_idx = [i for i, n in enumerate(feat_names)
          if any(n.endswith(f'_{f}') for f in FD_FEAT) and n.startswith('ch')]
cc_idx = [i for i, n in enumerate(feat_names) if n.startswith('cc_')]

print(f"Feature subsets — TD: {len(td_idx)}, FD: {len(fd_idx)}, CC: {len(cc_idx)}")
assert len(td_idx) == 36, f"Expected 36 TD features, got {len(td_idx)}"
assert len(fd_idx) == 24, f"Expected 24 FD features, got {len(fd_idx)}"
assert len(cc_idx) == 9,  f"Expected 9 CC features, got {len(cc_idx)}"

X_td = X[:, td_idx]
X_fd = X[:, fd_idx]
X_cc = X[:, cc_idx]

# --- Train specialist LDAs with out-of-fold stacking -------------------------
gkf = GroupKFold(n_splits=min(5, len(np.unique(trial_ids))))

print("Training specialist LDAs (out-of-fold for stacking)...")
lda_td = LinearDiscriminantAnalysis()
lda_fd = LinearDiscriminantAnalysis()
lda_cc = LinearDiscriminantAnalysis()

oof_td = cross_val_predict(lda_td, X_td, y, cv=gkf, groups=trial_ids, method='predict_proba')
oof_fd = cross_val_predict(lda_fd, X_fd, y, cv=gkf, groups=trial_ids, method='predict_proba')
oof_cc = cross_val_predict(lda_cc, X_cc, y, cv=gkf, groups=trial_ids, method='predict_proba')

# Specialist CV accuracy (for diagnostics)
for name, mdl, Xs in [('LDA_TD', lda_td, X_td),
                       ('LDA_FD', lda_fd, X_fd),
                       ('LDA_CC', lda_cc, X_cc)]:
    sc = cross_val_score(mdl, Xs, y, cv=gkf, groups=trial_ids)
    print(f"  {name}: {sc.mean()*100:.1f}% ± {sc.std()*100:.1f}%")

# --- Train meta-LDA on out-of-fold outputs ------------------------------------
X_meta = np.hstack([oof_td, oof_fd, oof_cc])   # (n_samples, 3*n_cls = 15)
meta_lda = LinearDiscriminantAnalysis()
meta_sc = cross_val_score(meta_lda, X_meta, y, cv=gkf, groups=trial_ids)
print(f"  Meta-LDA: {meta_sc.mean()*100:.1f}% ± {meta_sc.std()*100:.1f}%")

# Fit all models on full dataset for deployment
lda_td.fit(X_td, y)
lda_fd.fit(X_fd, y)
lda_cc.fit(X_cc, y)
meta_lda.fit(X_meta, y)

# --- Export all weights to C header ------------------------------------------
def lda_to_c_arrays(lda, name, feat_dim, n_cls, label_names, class_order):
    """Generate C array strings for LDA weights and intercepts.

    NOTE: sklearn LDA.coef_ for multi-class has shape (n_classes-1, n_features)
    when using SVD solver. If so, we use decision_function and re-derive weights.
    """
    coef = lda.coef_
    intercept = lda.intercept_

    if coef.shape[0] != n_cls:
        # SVD solver returns (n_cls-1, n_feat); sklearn handles this internally
        # via scalings_. We refit with 'lsqr' solver to get full coef matrix.
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA2
        lda2 = LDA2(solver='lsqr')
        # We can't refit here (no data) so just warn and pad with zeros
        print(f"  WARNING: {name} coef_ shape {coef.shape} != ({n_cls}, {feat_dim}). "
              f"Padding with zeros. Refit with solver='lsqr' for full matrix.")
        padded = np.zeros((n_cls, feat_dim))
        padded[:coef.shape[0]] = coef
        coef = padded
        padded_i = np.zeros(n_cls)
        padded_i[:intercept.shape[0]] = intercept
        intercept = padded_i

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
out_path = Path(__file__).parent / 'EMG_Arm/src/core/model_weights_ensemble.h'
out_path.parent.mkdir(parents=True, exist_ok=True)

td_offset = min(td_idx) if td_idx else 0
fd_offset = min(fd_idx) if fd_idx else 0
cc_offset = min(cc_idx) if cc_idx else 0

with open(out_path, 'w') as f:
    f.write("// Auto-generated by train_ensemble.py — do not edit\n")
    f.write("#pragma once\n\n")
    f.write("// Pull MODEL_NUM_CLASSES, MODEL_NUM_FEATURES, MODEL_CLASS_NAMES from\n")
    f.write("// model_weights.h to avoid redefinition conflicts.\n")
    f.write('#include "model_weights.h"\n\n')
    f.write(f"#define ENSEMBLE_PER_CH_FEATURES 20\n\n")
    f.write(f"#define TD_FEAT_OFFSET  {td_offset}\n")
    f.write(f"#define TD_NUM_FEATURES {len(td_idx)}\n")
    f.write(f"#define FD_FEAT_OFFSET  {fd_offset}\n")
    f.write(f"#define FD_NUM_FEATURES {len(fd_idx)}\n")
    f.write(f"#define CC_FEAT_OFFSET  {cc_offset}\n")
    f.write(f"#define CC_NUM_FEATURES {len(cc_idx)}\n")
    f.write(f"#define META_NUM_INPUTS (3 * MODEL_NUM_CLASSES)\n\n")

    f.write("// Feature index arrays for gather operations (TD and FD are non-contiguous)\n")
    f.write(f"// TD indices: {td_idx}\n")
    f.write(f"// FD indices: {fd_idx}\n")
    f.write(f"// CC indices: {cc_idx}\n\n")

    f.write(lda_to_c_arrays(lda_td,  'LDA_TD',  len(td_idx), n_cls, label_names, class_order))
    f.write('\n\n')
    f.write(lda_to_c_arrays(lda_fd,  'LDA_FD',  len(fd_idx), n_cls, label_names, class_order))
    f.write('\n\n')
    f.write(lda_to_c_arrays(lda_cc,  'LDA_CC',  len(cc_idx), n_cls, label_names, class_order))
    f.write('\n\n')
    f.write(lda_to_c_arrays(meta_lda, 'META_LDA', 3 * n_cls, n_cls, label_names, class_order))
    f.write('\n')

print(f"Exported ensemble weights to {out_path}")
print(f"Total weight storage: "
      f"{(len(td_idx) + len(fd_idx) + len(cc_idx) + 3*n_cls) * n_cls * 4} bytes float32")

# --- Also save sklearn models for laptop-side inference ----------------------
import joblib
ensemble_bundle = {
    'lda_td': lda_td,
    'lda_fd': lda_fd,
    'lda_cc': lda_cc,
    'meta_lda': meta_lda,
    'td_idx': td_idx,
    'fd_idx': fd_idx,
    'cc_idx': cc_idx,
    'label_names': label_names,
}
ensemble_joblib = Path(__file__).parent / 'models' / 'emg_ensemble.joblib'
ensemble_joblib.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(ensemble_bundle, ensemble_joblib)
print(f"Saved laptop ensemble model to {ensemble_joblib}")
