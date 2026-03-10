"""
EMG Signal Analysis Script (STANDALONE - NOT PRODUCTION CODE)
==============================================================

!! WARNING !!
This script is for OFFLINE ANALYSIS AND VISUALIZATION ONLY.
It is NOT part of the training or inference pipeline.

DO NOT use this script's outputs for:
- Model training
- Feature extraction thresholds
- Production inference

The thresholds defined here (ZC_THRESHOLD_PERCENT, SSC_THRESHOLD_PERCENT)
are DIFFERENT from production values in:
- learning_data_collection.py (production: 0.1, 0.1)
- model_weights.h (production: 0.1, 0.1)

This script uses higher thresholds (0.7, 0.6) for visualization clarity,
which would produce INCORRECT features if used for training/inference.

To analyze collected sessions:
1. Update HDF5_PATH to your session file
2. Run: python learning_emg_filtering.py
3. View the generated plots
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import butter, sosfiltfilt

# =============================================================================
# CONFIGURABLE PARAMETERS (FOR VISUALIZATION ONLY - NOT PRODUCTION VALUES!)
# =============================================================================
# These thresholds are intentionally different from production (0.1, 0.1)
# to provide cleaner visualizations. DO NOT use for training/inference.
ZC_THRESHOLD_PERCENT = 0.7  # Zero Crossing threshold (VISUALIZATION ONLY)
SSC_THRESHOLD_PERCENT = 0.6  # Slope Sign Change threshold (VISUALIZATION ONLY)

# =============================================================================
# LOAD DATA FROM GUI's HDF5 FORMAT
# =============================================================================
# Update this path to your collected session file
HDF5_PATH = "collected_data\latency_fix_106_20260127_200043.hdf5"

file = h5py.File(HDF5_PATH, "r")

# Print HDF5 structure for debugging
print("HDF5 Structure:")
def print_tree(name, obj):
    print(f"  {name}")
file.visititems(print_tree)
print()

# Load metadata
fs = file.attrs['sampling_rate']  # Sampling rate in Hz
n_channels = file.attrs['num_channels']
window_size_ms = file.attrs['window_size_ms']
print(f"Sampling rate: {fs} Hz")
print(f"Channels: {n_channels}")
print(f"Window size: {window_size_ms} ms")

# Load windowed EMG data: shape (n_windows, samples_per_window, channels)
emg_windows = file['windows/emg_data'][:]
labels = file['windows/labels'][:]
start_times = file['windows/start_times'][:]
end_times = file['windows/end_times'][:]

print(f"Windows shape: {emg_windows.shape}")
print(f"Labels: {len(labels)} (unique: {np.unique(labels)})")

# Flatten windows to continuous signal for filtering analysis
# Shape: (n_windows * samples_per_window, channels)
n_windows, samples_per_window, _ = emg_windows.shape
emg = emg_windows.reshape(-1, n_channels)
print(f"Flattened EMG shape: {emg.shape}")

# Reconstruct time vector (assumes continuous recording)
total_samples = emg.shape[0]
time = np.arange(total_samples) / fs
print(f"Time range: {time[0]:.2f}s to {time[-1]:.2f}s")

# =============================================================================
# PLOT RAW EMG DATA (before filtering)
# =============================================================================
print("\nPlotting raw EMG data...")

# Color map for channels
channel_colors = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffe66d']

# Map window indices to actual time in the flattened data
# Window i starts at sample (i * samples_per_window), which is time (i * samples_per_window / fs)
window_time_in_data = np.arange(len(labels)) * samples_per_window / fs

# Compute global Y range across all channels for consistent comparison
emg_global_min = emg.min()
emg_global_max = emg.max()
emg_margin = (emg_global_max - emg_global_min) * 0.05  # 5% margin
emg_ylim = (emg_global_min - emg_margin, emg_global_max + emg_margin)
print(f"Global EMG range: {emg_global_min:.1f} to {emg_global_max:.1f} mV")

# Full session plot with shared Y-axis
fig_raw, axes_raw = plt.subplots(n_channels, 1, figsize=(14, 2.5 * n_channels), sharex=True, sharey=True)
if n_channels == 1:
    axes_raw = [axes_raw]

for ch in range(n_channels):
    ax = axes_raw[ch]

    # Plot raw EMG signal
    ax.plot(time, emg[:, ch], linewidth=0.3, color=channel_colors[ch % len(channel_colors)], alpha=0.8)
    ax.set_ylabel(f'Ch {ch}\n(mV)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(emg_ylim)  # Same Y range for all channels

# Add gesture markers AFTER plotting
y_min, y_max = emg_ylim
y_text = y_min + (y_max - y_min) * 0.85  # Position text at 85% height
for ch in range(n_channels):
    ax = axes_raw[ch]

    # Add gesture transition markers using window-aligned times
    prev_label = None
    for i, lbl in enumerate(labels):
        lbl_str = lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl
        t = window_time_in_data[i]  # Time in the flattened data

        if lbl_str != prev_label:
            if 'open' in lbl_str:
                color = 'cyan'
            elif 'fist' in lbl_str:
                color = 'blue'
            elif 'hook' in lbl_str:
                color = 'orange'
            elif 'thumb' in lbl_str:
                color = 'green'
            elif 'rest' in lbl_str:
                color = 'gray'
            else:
                color = 'red'

            ax.axvline(t, color=color, linestyle='--', alpha=0.7, linewidth=1)

            # Only add text label on first channel to avoid clutter
            if ch == 0 and lbl_str != 'rest':
                ax.text(t + 0.2, y_text, lbl_str, fontsize=8,
                       color=color, rotation=0, ha='left', va='top',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        prev_label = lbl_str

axes_raw[-1].set_xlabel('Time (s)', fontsize=11)
fig_raw.suptitle('Raw EMG Signal (All Channels)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Zoomed view of first 2 seconds to see waveform detail
zoom_duration = 2.0
zoom_samples = int(zoom_duration * fs)

if total_samples > zoom_samples:
    fig_zoom, axes_zoom = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True, sharey=True)
    if n_channels == 1:
        axes_zoom = [axes_zoom]

    for ch in range(n_channels):
        ax = axes_zoom[ch]
        ax.plot(time[:zoom_samples], emg[:zoom_samples, ch],
                linewidth=0.5, color=channel_colors[ch % len(channel_colors)])
        ax.set_ylabel(f'Ch {ch}\n(mV)', fontsize=10)
        ax.set_ylim(emg_ylim)  # Same Y range as full plot
        ax.grid(True, alpha=0.3)

    axes_zoom[-1].set_xlabel('Time (s)', fontsize=11)
    fig_zoom.suptitle(f'Raw EMG Signal (First {zoom_duration}s - Zoomed)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 1) pick one channel
emg_ch = emg[:, 0].astype(np.float32)

# 2) drop the initial transient (recommended)
drop_s = 0.5
drop = int(drop_s * fs)
emg_ch = emg_ch[drop:]
time_ch = time[drop:]

# 3) design + apply bandpass
low, high = 20.0, 450.0
sos = butter(4, [low/(0.5*fs), high/(0.5*fs)], btype="bandpass", output="sos")
emg_bp = sosfiltfilt(sos, emg_ch)

# 4) RMS envelope extraction
# Step A: Define window size (150ms)
window_ms = 150
window_samples = int(window_ms / 1000 * fs)  # Convert ms to samples
print(f"Window: {window_ms}ms = {window_samples} samples")

# Step B: Square the bandpassed signal
emg_squared = emg_bp ** 2

# Step C: Moving average via convolution (rectangular kernel, normalized)
kernel = np.ones(window_samples) / window_samples
emg_mean_squared = np.convolve(emg_squared, kernel, mode='same')

# Step D: Square root to complete RMS
emg_envelope = np.sqrt(emg_mean_squared)

# =============================================================================
# TIME-DOMAIN FEATURE FUNCTIONS (RMS, WL, ZC, SSC)
# Computed on band-passed EMG, not RMS envelope.
# Designed for easy porting to embedded C.
# =============================================================================

def compute_rms(x):
    """Root Mean Square: sqrt(mean(x^2))"""
    return np.sqrt(np.mean(x ** 2))


def compute_wl(x):
    """Waveform Length: sum of absolute differences between consecutive samples."""
    return np.sum(np.abs(np.diff(x)))


def compute_zc(x, threshold):
    """Zero Crossings: count of sign changes where amplitude change exceeds threshold."""
    sign_change = x[:-1] * x[1:] < 0
    amp_diff = np.abs(np.diff(x)) > threshold
    return np.sum(sign_change & amp_diff)


def compute_ssc(x, threshold):
    """Slope Sign Changes: count of slope direction reversals exceeding threshold."""
    diff_left = x[1:-1] - x[:-2]
    diff_right = x[1:-1] - x[2:]
    return np.sum(diff_left * diff_right > threshold)


def compute_all_features_windowed(x, window_len, threshold_zc, threshold_ssc):
    """
    Compute RMS, WL, ZC, SSC using non-overlapping windows.
    Returns arrays of feature values, one per window (for ML).
    """
    n_samples = len(x)
    n_windows = n_samples // window_len
    x_trim = x[:n_windows * window_len]
    x_win = x_trim.reshape(n_windows, window_len)

    rms = np.sqrt(np.mean(x_win ** 2, axis=1))
    wl = np.sum(np.abs(np.diff(x_win, axis=1)), axis=1)

    sign_change = x_win[:, :-1] * x_win[:, 1:] < 0
    amp_diff = np.abs(np.diff(x_win, axis=1)) > threshold_zc
    zc = np.sum(sign_change & amp_diff, axis=1)

    diff_left = x_win[:, 1:-1] - x_win[:, :-2]
    diff_right = x_win[:, 1:-1] - x_win[:, 2:]
    ssc = np.sum(diff_left * diff_right > threshold_ssc, axis=1)

    return rms, wl, zc, ssc


# =============================================================================
# COMPUTE FEATURES FOR ALL CHANNELS
# =============================================================================

all_features = {}  # Non-overlapping windows - for ML

for ch in range(n_channels):
    emg_ch_i = emg[drop:, ch].astype(np.float32)
    emg_bp_i = sosfiltfilt(sos, emg_ch_i)

    signal_rms_i = np.sqrt(np.mean(emg_bp_i ** 2))
    threshold_zc_i = ZC_THRESHOLD_PERCENT * signal_rms_i
    threshold_ssc_i = (SSC_THRESHOLD_PERCENT * signal_rms_i) ** 2

    # Non-overlapping windowed features (for ML)
    rms_i, wl_i, zc_i, ssc_i = compute_all_features_windowed(
        emg_bp_i, window_samples, threshold_zc_i, threshold_ssc_i
    )
    all_features[ch] = {'rms': rms_i, 'wl': wl_i, 'zc': zc_i, 'ssc': ssc_i}

# Time vector for windowed features
n_feat_windows = len(all_features[0]['rms'])
window_centers = np.arange(n_feat_windows) * window_samples + window_samples // 2
time_windows = time_ch[window_centers]

print(f"\nComputed features for {n_channels} channels")
print(f"Windows per channel (non-overlapping): {n_feat_windows}")
print(f"Window size: {window_samples} samples ({window_ms} ms)")

# 5) Use embedded gesture labels from HDF5
# Labels are per-window, aligned with emg_windows
unique_labels = np.unique(labels)
print(f"\nUnique gestures in session: {unique_labels}")

# Map labels to time in the flattened data (same as raw plot)
# Each original window i maps to time (i * samples_per_window / fs) in the flattened data
# But we dropped `drop` samples, so adjust accordingly
label_times_in_data = np.arange(len(labels)) * samples_per_window / fs
print(f"Data duration: {time[-1]:.2f}s ({len(labels)} windows)")

# Color function for markers
def get_gesture_color(name):
    """Assign colors to gesture types."""
    name_str = name.decode('utf-8') if isinstance(name, bytes) else name
    if 'open' in name_str:
        return 'cyan'
    elif 'fist' in name_str:
        return 'blue'
    elif 'hook' in name_str:
        return 'orange'
    elif 'thumb' in name_str:
        return 'green'
    elif 'rest' in name_str:
        return 'gray'
    return 'red'

# =============================================================================
# 6) PLOT ALL FEATURES (RMS, WL, ZC, SSC) FOR ALL CHANNELS
# =============================================================================

feature_names = ['rms', 'wl', 'zc', 'ssc']
feature_titles = ['RMS Envelope', 'Waveform Length (WL)', 'Zero Crossings (ZC)', 'Slope Sign Changes (SSC)']
feature_colors = ['red', 'blue', 'green', 'purple']
feature_ylabels = ['Amplitude (mV)', 'WL (a.u.)', 'Count', 'Count']

# Determine subplot grid based on channel count
if n_channels <= 4:
    n_rows, n_cols = 2, 2
elif n_channels <= 9:
    n_rows, n_cols = 3, 3
else:
    n_rows, n_cols = 4, 4

# Plot entire session for each feature
for feat_idx, feat_name in enumerate(feature_names):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # Compute global Y range for this feature across all channels
    all_feat_vals = np.concatenate([all_features[ch][feat_name] for ch in range(n_channels)])
    feat_min, feat_max = all_feat_vals.min(), all_feat_vals.max()
    feat_margin = (feat_max - feat_min) * 0.05 if feat_max > feat_min else 1
    feat_ylim = (feat_min - feat_margin, feat_max + feat_margin)

    for ch in range(n_channels):
        ax = axes[ch]

        # Plot windowed feature over time
        feat_data = all_features[ch][feat_name]
        ax.plot(time_windows, feat_data, linewidth=0.8, color=feature_colors[feat_idx])
        ax.set_title(f"Channel {ch}", fontsize=10)
        ax.set_ylim(feat_ylim)  # Same Y range for all channels
        ax.grid(True, alpha=0.3)

        # Add gesture transition markers from labels
        prev_label = None
        for i, lbl in enumerate(labels):
            lbl_str = lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl
            t = label_times_in_data[i]  # Time in flattened data
            if lbl_str != prev_label and lbl_str != 'rest':
                # Only show markers within the time_windows range
                if t <= time_windows[-1]:
                    color = get_gesture_color(lbl)
                    ax.axvline(t, color=color, linestyle='--', alpha=0.6, linewidth=1)
            prev_label = lbl_str

    # Hide unused subplots
    for ch in range(n_channels, len(axes)):
        axes[ch].set_visible(False)

    # Legend for gesture colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan', linestyle='--', label='Open'),
        Line2D([0], [0], color='blue', linestyle='--', label='Fist'),
        Line2D([0], [0], color='orange', linestyle='--', label='Hook Em'),
        Line2D([0], [0], color='green', linestyle='--', label='Thumbs Up'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9)

    fig.suptitle(f"{feature_titles[feat_idx]} - All Channels", fontsize=14)
    fig.supxlabel("Time (s)")
    fig.supylabel(feature_ylabels[feat_idx])
    plt.tight_layout()
    plt.show()

# =============================================================================
# SUMMARY: Feature statistics across all channels
# =============================================================================
print("\n" + "=" * 60)
print(f"FEATURE SUMMARY ({n_channels} channels, {n_feat_windows} windows each)")
print("=" * 60)
for feat_name in ['rms', 'wl', 'zc', 'ssc']:
    all_vals = np.concatenate([all_features[ch][feat_name] for ch in range(n_channels)])
    print(f"{feat_name.upper():4s} | min: {all_vals.min():10.4f} | max: {all_vals.max():10.4f} | "
          f"mean: {all_vals.mean():10.4f} | std: {all_vals.std():10.4f}")

# Close the HDF5 file
file.close()
print("\n[Done] HDF5 file closed.")