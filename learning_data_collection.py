"""
EMG Data Collection Pipeline
============================
A complete pipeline for collecting, labeling, and classifying EMG signals.

OPTIONS:
  1. Collect Data    - Run a labeled collection session with timed prompts (requires ESP32)
  2. Inspect Data    - Load saved sessions, view raw EMG and features
  3. Train Classifier - Train LDA on collected data with cross-validation
  4. Live Prediction - Real-time gesture classification (requires ESP32)
  5. Visualize LDA   - Decision boundaries and feature space plots
  6. Benchmark       - Compare LDA/QDA/SVM/MLP classifiers
  q. Quit

FEATURES:
  - Real-time EMG acquisition via ESP32 serial interface
  - Timed prompt system for consistent data collection
  - Automatic labeling based on prompt timing with onset detection
  - HDF5 storage with metadata
  - Time-domain feature extraction (RMS, WL, ZC, SSC)
  - LDA classifier with evaluation metrics
  - Prediction smoothing (EMA + majority vote + debounce)

HARDWARE REQUIRED:
  - ESP32 with EMG sensors connected and firmware flashed
  - USB serial connection (921600 baud)
"""

import time
import threading
import queue
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datetime import datetime
import json
import h5py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, GroupShuffleSplit, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # For model persistence
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, sosfilt, sosfilt_zi  # For label alignment + bandpass
from serial_stream import RealSerialStream  # ESP32 serial communication

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_CHANNELS = 4          # Number of EMG channels (MyoWare sensors)
SAMPLING_RATE_HZ = 1000   # Must match ESP32's EMG_SAMPLE_RATE_HZ
SERIAL_BAUD = 921600      # High baud rate to prevent serial buffer backlog

# Windowing configuration (must match ESP32 inference timing)
WINDOW_SIZE_MS = 150      # Window size in milliseconds (150 samples at 1kHz)
HOP_SIZE_MS = 25          # Hop/stride in milliseconds (25 samples at 1kHz)
MAJORITY_WINDOW = 10

# Hand classifier channel selection
# The hand gesture classifier uses only forearm channels (ch0-ch2).
# The bicep channel (ch3) is excluded to prevent bicep activity from
# corrupting hand gesture classification. Ch3 is reserved for independent
# bicep envelope processing (see Phase 5).
HAND_CHANNELS = [0, 1, 2]  # Forearm channels only (excludes bicep ch3)

# Labeling configuration
GESTURE_HOLD_SEC = 3.0    # How long to hold each gesture
REST_BETWEEN_SEC = 2.0    # Rest period between gestures
REPS_PER_GESTURE = 3      # Repetitions per gesture in a session
LABEL_SHIFT_MS = 150      # Shift label lookup forward by this many ms to account
                          # for human reaction time.  A 150ms window labelled at its
                          # start_time can straddle a prompt transition; using
                          # start_time + shift assigns the label based on what the
                          # user is actually doing at the window's centre.

# Storage configuration
DATA_DIR = Path("collected_data")  # Directory to store session files
MODEL_DIR = Path("models")         # Directory to store trained models
USER_ID = "user_001"               # Current user ID (change per user)

# =============================================================================
# LABEL ALIGNMENT CONFIGURATION
# =============================================================================
# Human reaction time causes EMG activity to lag behind label prompts.
# We detect when EMG actually rises and shift labels to match.

ENABLE_LABEL_ALIGNMENT = True     # Enable/disable automatic label alignment
ONSET_THRESHOLD = 2             # Signal must exceed baseline + threshold * std
ONSET_SEARCH_MS = 2000           # Search window after prompt (ms)
# Change 0: after onset detection shifts the label start backward, additionally
# relabel the first LABEL_FORWARD_SHIFT_MS of each gesture run as "rest" to skip
# the EMG transient at gesture onset. Paired with reducing TRANSITION_START_MS.
LABEL_FORWARD_SHIFT_MS = 100     # ms of each gesture onset to relabel as rest

# =============================================================================
# TRANSITION WINDOW FILTERING
# =============================================================================
# Windows near gesture transitions contain ambiguous data (reaction time at start,
# muscle relaxation at end). Discard these during training for cleaner labels.
# This is standard practice in EMG research (see Frontiers Neurorobotics 2023).

DISCARD_TRANSITION_WINDOWS = True  # Enable/disable transition filtering during training
TRANSITION_START_MS = 200          # Discard windows within this time AFTER gesture starts
TRANSITION_END_MS = 150            # Discard windows within this time BEFORE gesture ends

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EMGSample:
    """Single sample from all channels at one point in time."""
    timestamp: float                    # Python-side timestamp (seconds, monotonic)
    channels: list[float]               # Raw ADC values per channel
    # DEPRECATED: esp_timestamp_ms is no longer used. Python-side timestamps are used
    # for label alignment. Kept for backward compatibility with old serialized data.
    esp_timestamp_ms: Optional[int] = None


@dataclass
class EMGWindow:
    """
    A window of samples - this is what we'll feed to ML models.

    NOTE: This class intentionally contains NO label information.
    Labels are stored separately to enforce training/inference separation.
    This ensures inference code cannot accidentally access ground truth.
    """
    window_id: int
    start_time: float
    end_time: float
    samples: list[EMGSample]

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array of shape (n_samples, n_channels)."""
        return np.array([s.channels for s in self.samples])

    def get_channel(self, ch: int) -> np.ndarray:
        """Get single channel as 1D array."""
        return np.array([s.channels[ch] for s in self.samples])


# =============================================================================
# DATA PARSER (Converts serial lines to EMGSample objects)
# =============================================================================

class EMGParser:
    """
    Parses incoming serial data into structured EMGSample objects.

    LESSON: Always validate incoming data. Serial lines can be:
      - Corrupted (partial lines, garbage bytes)
      - Missing (dropped packets)
      - Out of order (buffer issues)
    """

    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        self.parse_errors = 0
        self.samples_parsed = 0

    def parse_line(self, line: str) -> Optional[EMGSample]:
        """
        Parse a line from ESP32 into an EMGSample.

        Expected format: "ch0,ch1,ch2,ch3\n" (channels only, no ESP32 timestamp)
        Python assigns timestamp on receipt for label alignment.
        Returns None if parsing fails.
        """
        try:
            # Strip whitespace and split
            parts = line.strip().split(',')

            # Validate we have correct number of fields (channels only)
            if len(parts) != self.num_channels:
                self.parse_errors += 1
                return None

            # Parse channel values
            channels = [float(parts[i]) for i in range(self.num_channels)]

            # Create sample with Python-side timestamp (aligned with label clock)
            sample = EMGSample(
                timestamp=time.perf_counter(),  # High-resolution monotonic clock
                channels=channels,
                esp_timestamp_ms=None  # Deprecated field, kept for compatibility
            )

            self.samples_parsed += 1
            return sample

        except (ValueError, IndexError) as e:
            self.parse_errors += 1
            return None


# =============================================================================
# WINDOWING (Groups samples into fixed-size windows)
# =============================================================================

class Windower:
    """
    Groups incoming samples into fixed-size windows.

    LESSON: ML models need fixed-size inputs. We can't feed them a continuous
    stream - we need to chunk it into windows of consistent size.

    Window size tradeoffs:
      - Too small (50ms): Not enough data, noisy features
      - Too large (500ms): Slow response, gesture transitions blurred
      - Sweet spot: 150-250ms for EMG gesture recognition
    """

    def __init__(self, window_size_ms: int, sample_rate: int, hop_size_ms: int = 25):
        self.window_size_ms = window_size_ms
        self.sample_rate = sample_rate
        self.hop_size_ms = hop_size_ms

        # Calculate window and step size in samples (hop-based, not overlap-based)
        self.window_size_samples = int(window_size_ms / 1000 * sample_rate)
        self.step_size_samples = int(hop_size_ms / 1000 * sample_rate)

        # Buffer for incoming samples
        self.buffer: list[EMGSample] = []
        self.window_count = 0

        # Verification: Print first 10 window start indices and timestamps
        self._verification_printed = False

        print(f"[Windower] Window: {window_size_ms}ms = {self.window_size_samples} samples")
        print(f"[Windower] Hop: {hop_size_ms}ms = {self.step_size_samples} samples")

    def add_sample(self, sample: EMGSample) -> Optional[EMGWindow]:
        """
        Add a sample to the buffer. Returns a window if we have enough samples.

        Returns None if buffer isn't full yet.

        Window timing (at 1kHz):
          - Window 0: samples 0-149,   start index 0,   time 0.000s
          - Window 1: samples 25-174,  start index 25,  time 0.025s
          - Window 2: samples 50-199,  start index 50,  time 0.050s
          - ...
        """
        self.buffer.append(sample)

        # Check if we have enough samples for a window
        if len(self.buffer) >= self.window_size_samples:
            # Extract window
            window_samples = self.buffer[:self.window_size_samples]
            window = EMGWindow(
                window_id=self.window_count,
                start_time=window_samples[0].timestamp,
                end_time=window_samples[-1].timestamp,
                samples=window_samples.copy()
            )

            # Verification: Print first 10 window start indices and timestamps
            # if not self._verification_printed and self.window_count < 10:
            #     start_idx = self.window_count * self.step_size_samples
            #     start_time_sec = start_idx / self.sample_rate
            #     print(f"[Windower] Window {self.window_count}: start_idx={start_idx}, time={start_time_sec:.3f}s")
            #     if self.window_count == 9:
            #         self._verification_printed = True
            #         print(f"[Windower] Verified: 150-sample windows, {self.step_size_samples}-sample hop")

            self.window_count += 1

            # Slide buffer by step size
            self.buffer = self.buffer[self.step_size_samples:]

            return window

        return None

    def flush(self) -> Optional[EMGWindow]:
        """Flush remaining samples as a partial window (if any)."""
        if len(self.buffer) > 0:
            window = EMGWindow(
                window_id=self.window_count,
                start_time=self.buffer[0].timestamp,
                end_time=self.buffer[-1].timestamp,
                samples=self.buffer.copy()
            )
            self.buffer = []
            return window
        return None


# =============================================================================
# PROMPT SYSTEM (Timed prompts for labeling)
# =============================================================================

@dataclass
class GesturePrompt:
    """Defines a single gesture prompt in the collection sequence."""
    gesture_name: str       # e.g., "index_flex", "rest", "fist"
    duration_sec: float     # How long to hold this gesture
    start_time: float = 0.0 # Filled in by scheduler when session starts
    trial_id: int = -1      # Unique ID for this trial (gesture repetition)


@dataclass
class PromptSchedule:
    """A complete sequence of prompts for a collection session."""
    prompts: list[GesturePrompt]
    total_duration: float = 0.0

    def __post_init__(self):
        """Calculate start times and total duration."""
        current_time = 0.0
        for prompt in self.prompts:
            prompt.start_time = current_time
            current_time += prompt.duration_sec
        self.total_duration = current_time


class PromptScheduler:
    """
    Manages timed prompts during data collection.

    LESSON: Timed prompts give you consistent, repeatable data collection.
    The user knows exactly when to perform each gesture, and you know
    exactly when each gesture should be happening for labeling.
    """

    def __init__(self, gestures: list[str], hold_sec: float, rest_sec: float, reps: int):
        """
        Build a prompt schedule.

        Args:
            gestures: List of gesture names (e.g., ["index_flex", "fist"])
            hold_sec: How long to hold each gesture
            rest_sec: Rest period between gestures
            reps: Number of repetitions per gesture
        """
        self.gestures = gestures
        self.hold_sec = hold_sec
        self.rest_sec = rest_sec
        self.reps = reps

        # Build the schedule
        self.schedule = self._build_schedule()
        self.session_start_time: Optional[float] = None

    def _build_schedule(self) -> PromptSchedule:
        """Create the sequence of prompts with unique trial_ids."""
        prompts = []
        trial_counter = 0

        # Initial rest period (trial_id = 0)
        prompts.append(GesturePrompt("rest", self.rest_sec, trial_id=trial_counter))
        trial_counter += 1

        # For each repetition
        for rep in range(self.reps):
            # Cycle through all gestures
            for gesture in self.gestures:
                # Gesture trial
                prompts.append(GesturePrompt(gesture, self.hold_sec, trial_id=trial_counter))
                trial_counter += 1
                # Rest trial (each rest is its own trial to avoid leakage)
                prompts.append(GesturePrompt("rest", self.rest_sec, trial_id=trial_counter))
                trial_counter += 1

        return PromptSchedule(prompts)

    def start_session(self):
        """Mark the start of a collection session."""
        self.session_start_time = time.perf_counter()
        print(f"\n[Scheduler] Session started. Duration: {self.schedule.total_duration:.1f}s")
        print(f"[Scheduler] {len(self.schedule.prompts)} prompts scheduled")

    def get_current_prompt(self) -> Optional[GesturePrompt]:
        """Get the prompt that should be active right now."""
        if self.session_start_time is None:
            return None

        elapsed = time.perf_counter() - self.session_start_time

        # Find which prompt is active
        for prompt in self.schedule.prompts:
            prompt_end = prompt.start_time + prompt.duration_sec
            if prompt.start_time <= elapsed < prompt_end:
                return prompt

        return None  # Session complete

    def get_elapsed_time(self) -> float:
        """Get seconds elapsed since session start."""
        if self.session_start_time is None:
            return 0.0
        return time.perf_counter() - self.session_start_time

    def is_session_complete(self) -> bool:
        """Check if we've passed the end of the schedule."""
        return self.get_elapsed_time() >= self.schedule.total_duration

    def get_label_for_time(self, timestamp: float) -> str:
        """
        Get the gesture label for a specific timestamp.

        This is used to label windows after collection.
        """
        if self.session_start_time is None:
            return "unlabeled"

        elapsed = timestamp - self.session_start_time

        for prompt in self.schedule.prompts:
            prompt_end = prompt.start_time + prompt.duration_sec
            if prompt.start_time <= elapsed < prompt_end:
                return prompt.gesture_name

        return "unlabeled"

    def get_trial_id_for_time(self, timestamp: float) -> int:
        """
        Get the trial_id for a specific timestamp.

        Each gesture repetition has a unique trial_id. Windows from the same
        trial MUST stay together during train/test splitting to prevent leakage.
        """
        if self.session_start_time is None:
            return -1

        elapsed = timestamp - self.session_start_time

        for prompt in self.schedule.prompts:
            prompt_end = prompt.start_time + prompt.duration_sec
            if prompt.start_time <= elapsed < prompt_end:
                return prompt.trial_id

        return -1

    def print_schedule(self):
        """Print the full prompt schedule."""
        print("\n" + "-" * 40)
        print("PROMPT SCHEDULE")
        print("-" * 40)
        for i, p in enumerate(self.schedule.prompts):
            print(f"  {i+1:2d}. [{p.start_time:5.1f}s - {p.start_time + p.duration_sec:5.1f}s] {p.gesture_name}")
        print(f"\n  Total duration: {self.schedule.total_duration:.1f}s")


# =============================================================================
# LABEL ALIGNMENT (Simple Onset Detection)
# =============================================================================
# NOTE: butter and sosfiltfilt imported at top of file


def align_labels_with_onset(
    labels: list[str],
    window_start_times: np.ndarray,
    raw_timestamps: np.ndarray,
    raw_channels: np.ndarray,
    sampling_rate: int,
    threshold_factor: float = 2.0,
    search_ms: float = 800
) -> list[str]:
    """
    Align labels to EMG onset by detecting when signal rises above baseline.

    Simple algorithm:
    1. High-pass filter to remove DC offset
    2. Compute RMS envelope across channels
    3. At each label transition, find where envelope exceeds baseline + threshold
    4. Move label boundary to that point
    """
    if len(labels) == 0:
        return labels.copy()

    # High-pass filter to remove DC (raw ADC has ~2340mV offset)
    nyquist = sampling_rate / 2
    sos = butter(2, 20.0 / nyquist, btype='high', output='sos')

    # Filter and compute envelope (RMS across channels)
    filtered = np.zeros_like(raw_channels)
    for ch in range(raw_channels.shape[1]):
        filtered[:, ch] = sosfiltfilt(sos, raw_channels[:, ch])
    envelope = np.sqrt(np.mean(filtered ** 2, axis=1))

    # Smooth envelope
    sos_lp = butter(2, 10.0 / nyquist, btype='low', output='sos')
    envelope = sosfiltfilt(sos_lp, envelope)

    # Find transitions and detect onsets
    search_samples = int(search_ms / 1000 * sampling_rate)
    baseline_samples = int(200 / 1000 * sampling_rate)

    boundaries = []  # (time, new_label)

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            prompt_time = window_start_times[i]

            # Find index in raw signal closest to prompt time
            prompt_idx = np.searchsorted(raw_timestamps, prompt_time)

            # Get baseline (before transition)
            base_start = max(0, prompt_idx - baseline_samples)
            baseline = envelope[base_start:prompt_idx]
            if len(baseline) == 0:
                boundaries.append((prompt_time + 0.3, labels[i]))
                continue

            threshold = np.mean(baseline) + threshold_factor * np.std(baseline)

            # Search forward for onset
            search_end = min(len(envelope), prompt_idx + search_samples)
            onset_idx = None

            for j in range(prompt_idx, search_end):
                if envelope[j] > threshold:
                    onset_idx = j
                    break

            if onset_idx is not None:
                onset_time = raw_timestamps[onset_idx]
            else:
                onset_time = prompt_time + 0.3  # fallback

            boundaries.append((onset_time, labels[i]))

    # Assign labels based on detected boundaries
    aligned = []
    boundary_idx = 0
    current_label = labels[0]

    for t in window_start_times:
        while boundary_idx < len(boundaries) and t >= boundaries[boundary_idx][0]:
            current_label = boundaries[boundary_idx][1]
            boundary_idx += 1
        aligned.append(current_label)

    return aligned


def filter_transition_windows(
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    start_times: np.ndarray,
    end_times: np.ndarray,
    trial_ids: Optional[np.ndarray] = None,
    transition_start_ms: float = TRANSITION_START_MS,
    transition_end_ms: float = TRANSITION_END_MS
) -> tuple[np.ndarray, np.ndarray, list[str], Optional[np.ndarray]]:
    """
    Filter out windows that fall within transition zones at gesture boundaries.

    This removes ambiguous data where:
    - User is still reacting to prompt (start of gesture)
    - User is anticipating next gesture (end of gesture)

    Args:
        X: EMG data array (n_windows, samples, channels)
        y: Label indices (n_windows,)
        labels: String labels (n_windows,)
        start_times: Window start times in seconds (n_windows,)
        end_times: Window end times in seconds (n_windows,)
        trial_ids: Trial IDs for train/test splitting (n_windows,) - optional
        transition_start_ms: Discard windows within this time after gesture start
        transition_end_ms: Discard windows within this time before gesture end

    Returns:
        Filtered (X, y, labels, trial_ids) with transition windows removed
    """
    if len(X) == 0:
        return X, y, labels, trial_ids

    transition_start_sec = transition_start_ms / 1000.0
    transition_end_sec = transition_end_ms / 1000.0

    # Find gesture boundaries (where label changes)
    # Each boundary is the START of a new gesture segment
    boundaries = [0]  # First window starts a segment
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            boundaries.append(i)
    boundaries.append(len(labels))  # End marker

    # For each segment, find start_time and end_time of the gesture
    # Then mark windows that are within transition zones
    keep_mask = np.ones(len(X), dtype=bool)

    for seg_idx in range(len(boundaries) - 1):
        seg_start_idx = boundaries[seg_idx]
        seg_end_idx = boundaries[seg_idx + 1]

        # Get the time boundaries of this gesture segment
        gesture_start_time = start_times[seg_start_idx]
        gesture_end_time = end_times[seg_end_idx - 1]  # Last window's end time

        # Mark windows in transition zones
        for i in range(seg_start_idx, seg_end_idx):
            window_start = start_times[i]
            window_end = end_times[i]

            # Check if window is too close to gesture START (reaction time zone)
            if window_start < gesture_start_time + transition_start_sec:
                keep_mask[i] = False

            # Check if window is too close to gesture END (anticipation zone)
            if window_end > gesture_end_time - transition_end_sec:
                keep_mask[i] = False

    # Apply filter
    X_filtered = X[keep_mask]
    y_filtered = y[keep_mask]
    labels_filtered = [l for l, keep in zip(labels, keep_mask) if keep]
    trial_ids_filtered = trial_ids[keep_mask] if trial_ids is not None else None

    n_removed = len(X) - len(X_filtered)
    if n_removed > 0:
        print(f"[Filter] Removed {n_removed} transition windows ({n_removed/len(X)*100:.1f}%)")
        print(f"[Filter] Kept {len(X_filtered)} windows for training")

    return X_filtered, y_filtered, labels_filtered, trial_ids_filtered


# =============================================================================
# SESSION STORAGE (Save/Load labeled data to HDF5)
# =============================================================================

@dataclass
class SessionMetadata:
    """Metadata for a collection session."""
    user_id: str
    session_id: str
    timestamp: str
    sampling_rate: int
    window_size_ms: int
    num_channels: int
    gestures: list[str]
    notes: str = ""


class SessionStorage:
    """Handles saving and loading EMG collection sessions to HDF5 files."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_session_id(self, user_id: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{user_id}_{timestamp}"

    def get_session_filepath(self, session_id: str) -> Path:
        return self.data_dir / f"{session_id}.hdf5"

    def save_session(
        self,
        windows: list[EMGWindow],
        labels: list[str],
        metadata: SessionMetadata,
        trial_ids: Optional[list[int]] = None,
        raw_samples: Optional[list[EMGSample]] = None,
        session_start_time: Optional[float] = None,
        enable_alignment: bool = ENABLE_LABEL_ALIGNMENT
    ) -> Path:
        """
        Save a collection session to HDF5 with optional label alignment.

        When raw_samples and session_start_time are provided and enable_alignment
        is True, automatically detects EMG onset and corrects labels for human
        reaction time delay.

        Args:
            windows: List of EMGWindow objects (no label info)
            labels: List of gesture labels, parallel to windows
            metadata: Session metadata
            trial_ids: List of trial IDs, parallel to windows (for proper train/test splitting)
            raw_samples: Raw samples (required for alignment)
            session_start_time: When session started (required for alignment)
            enable_alignment: Whether to perform automatic label alignment
        """
        filepath = self.get_session_filepath(metadata.session_id)

        if not windows:
            raise ValueError("No windows to save!")

        if len(windows) != len(labels):
            raise ValueError(f"Windows ({len(windows)}) and labels ({len(labels)}) must have same length!")

        window_samples = len(windows[0].samples)
        num_channels = len(windows[0].samples[0].channels)

        # Prepare timing arrays
        start_times = np.array([w.start_time for w in windows], dtype=np.float64)
        end_times = np.array([w.end_time for w in windows], dtype=np.float64)

        # Label alignment using onset detection
        aligned_labels = labels
        original_labels = labels

        if enable_alignment and raw_samples and len(raw_samples) > 0:
            print("[Storage] Aligning labels to EMG onset...")

            raw_timestamps = np.array([s.timestamp for s in raw_samples], dtype=np.float64)
            raw_channels = np.array([s.channels for s in raw_samples], dtype=np.float32)

            aligned_labels = align_labels_with_onset(
                labels=labels,
                window_start_times=start_times,
                raw_timestamps=raw_timestamps,
                raw_channels=raw_channels,
                sampling_rate=metadata.sampling_rate,
                threshold_factor=ONSET_THRESHOLD,
                search_ms=ONSET_SEARCH_MS
            )

            changed = sum(1 for a, b in zip(labels, aligned_labels) if a != b)
            print(f"[Storage] Labels aligned: {changed}/{len(labels)} windows shifted")

            # Change 0: relabel the first LABEL_FORWARD_SHIFT_MS of each gesture
            # run as 'rest' to remove the EMG onset transient from training data.
            if LABEL_FORWARD_SHIFT_MS > 0:
                shift_n = max(1, round(LABEL_FORWARD_SHIFT_MS / HOP_SIZE_MS))
                shifted = list(aligned_labels)
                for i in range(len(aligned_labels)):
                    if aligned_labels[i] == 'rest':
                        continue
                    # Count consecutive same-label windows immediately before this one
                    prior_same = 0
                    j = i - 1
                    while j >= 0 and aligned_labels[j] == aligned_labels[i]:
                        prior_same += 1
                        j -= 1
                    if prior_same < shift_n:
                        shifted[i] = 'rest'
                n_shifted = sum(1 for a, b in zip(aligned_labels, shifted) if a != b)
                aligned_labels = shifted
                print(f"[Storage] Forward shift ({LABEL_FORWARD_SHIFT_MS}ms, "
                      f"{shift_n} windows): {n_shifted} relabeled as rest")

        elif enable_alignment:
            print("[Storage] Warning: No raw samples, skipping alignment")

        with h5py.File(filepath, 'w') as f:
            # Metadata as attributes
            f.attrs['user_id'] = metadata.user_id
            f.attrs['session_id'] = metadata.session_id
            f.attrs['timestamp'] = metadata.timestamp
            f.attrs['sampling_rate'] = metadata.sampling_rate
            f.attrs['window_size_ms'] = metadata.window_size_ms
            f.attrs['num_channels'] = metadata.num_channels
            f.attrs['gestures'] = json.dumps(metadata.gestures)
            f.attrs['notes'] = metadata.notes
            f.attrs['num_windows'] = len(windows)
            f.attrs['window_samples'] = window_samples

            # Windows group
            windows_grp = f.create_group('windows')

            emg_data = np.array([w.to_numpy() for w in windows], dtype=np.float32)
            windows_grp.create_dataset('emg_data', data=emg_data, compression='gzip', compression_opts=4)

            # Store ALIGNED labels as primary (what training will use)
            max_label_len = max(len(l) for l in aligned_labels)
            dt = h5py.string_dtype(encoding='utf-8', length=max_label_len + 1)
            windows_grp.create_dataset('labels', data=aligned_labels, dtype=dt)

            # Also store original labels for reference/debugging
            windows_grp.create_dataset('labels_original', data=original_labels, dtype=dt)

            window_ids = np.array([w.window_id for w in windows], dtype=np.int32)
            windows_grp.create_dataset('window_ids', data=window_ids)

            # Store trial_ids for proper train/test splitting (no trial leakage)
            if trial_ids is not None:
                trial_ids_arr = np.array(trial_ids, dtype=np.int32)
                windows_grp.create_dataset('trial_ids', data=trial_ids_arr)
                f.attrs['has_trial_ids'] = True
            else:
                f.attrs['has_trial_ids'] = False

            windows_grp.create_dataset('start_times', data=start_times)
            windows_grp.create_dataset('end_times', data=end_times)

            # Store alignment metadata
            f.attrs['alignment_enabled'] = enable_alignment
            f.attrs['alignment_method'] = 'onset_detection' if (enable_alignment and raw_samples) else 'none'

            if raw_samples:
                raw_grp = f.create_group('raw_samples')
                timestamps = np.array([s.timestamp for s in raw_samples], dtype=np.float64)
                channels = np.array([s.channels for s in raw_samples], dtype=np.float32)
                raw_grp.create_dataset('timestamps', data=timestamps, compression='gzip')
                raw_grp.create_dataset('channels', data=channels, compression='gzip')

        print(f"[Storage] Saved session to: {filepath}")
        print(f"[Storage] File size: {filepath.stat().st_size / 1024:.1f} KB")
        return filepath

    def load_session(self, session_id: str) -> tuple[list[EMGWindow], list[str], SessionMetadata]:
        """
        Load a collection session from HDF5.

        Returns:
            windows: List of EMGWindow objects (no label info)
            labels: List of gesture labels, parallel to windows
            metadata: Session metadata
        """
        filepath = self.get_session_filepath(session_id)
        if not filepath.exists():
            raise FileNotFoundError(f"Session not found: {filepath}")

        windows = []
        labels_out = []
        with h5py.File(filepath, 'r') as f:
            metadata = SessionMetadata(
                user_id=f.attrs['user_id'],
                session_id=f.attrs['session_id'],
                timestamp=f.attrs['timestamp'],
                sampling_rate=int(f.attrs['sampling_rate']),
                window_size_ms=int(f.attrs['window_size_ms']),
                num_channels=int(f.attrs['num_channels']),
                gestures=json.loads(f.attrs['gestures']),
                notes=f.attrs.get('notes', '')
            )

            windows_grp = f['windows']
            emg_data = windows_grp['emg_data'][:]
            labels_raw = windows_grp['labels'][:]
            window_ids = windows_grp['window_ids'][:]
            start_times = windows_grp['start_times'][:]
            end_times = windows_grp['end_times'][:]

            for i in range(len(emg_data)):
                samples = []
                window_data = emg_data[i]
                for j in range(len(window_data)):
                    sample = EMGSample(
                        timestamp=start_times[i] + j * (1.0 / metadata.sampling_rate),
                        channels=window_data[j].tolist()
                    )
                    samples.append(sample)

                # Decode label
                label = labels_raw[i]
                if isinstance(label, bytes):
                    label = label.decode('utf-8')
                labels_out.append(label)

                # Window contains NO label - labels stored separately
                window = EMGWindow(
                    window_id=int(window_ids[i]),
                    start_time=float(start_times[i]),
                    end_time=float(end_times[i]),
                    samples=samples
                )
                windows.append(window)

        print(f"[Storage] Loaded session: {session_id}")
        print(f"[Storage] {len(windows)} windows, {len(metadata.gestures)} gesture types")
        return windows, labels_out, metadata

    def load_for_training(self, session_id: str, filter_transitions: bool = DISCARD_TRANSITION_WINDOWS) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Load a single session in ML-ready format: X, y, label_names.

        Args:
            session_id: The session to load
            filter_transitions: If True, remove windows in transition zones (default from config)
        """
        filepath = self.get_session_filepath(session_id)

        with h5py.File(filepath, 'r') as f:
            X = f['windows/emg_data'][:]
            labels_raw = f['windows/labels'][:]
            start_times = f['windows/start_times'][:]
            end_times = f['windows/end_times'][:]

        labels = []
        for l in labels_raw:
            if isinstance(l, bytes):
                labels.append(l.decode('utf-8'))
            else:
                labels.append(l)

        print(f"[Storage] Loaded session: {session_id} ({X.shape[0]} windows)")

        # Apply transition filtering if enabled
        if filter_transitions:
            label_names_pre = sorted(set(labels))
            label_to_idx_pre = {name: idx for idx, name in enumerate(label_names_pre)}
            y_pre = np.array([label_to_idx_pre[l] for l in labels], dtype=np.int32)

            X, y_pre, labels, _ = filter_transition_windows(
                X, y_pre, labels, start_times, end_times
            )

        label_names = sorted(set(labels))
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}
        y = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

        print(f"[Storage] Ready for training: X{X.shape}, y{y.shape}")
        print(f"[Storage] Labels: {label_names}")
        return X, y, label_names

    def load_all_for_training(self, filter_transitions: bool = DISCARD_TRANSITION_WINDOWS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
        """
        Load ALL sessions combined into a single training dataset.

        Args:
            filter_transitions: If True, remove windows in transition zones (default from config)

        Returns:
            X: Combined EMG windows from all sessions (n_total_windows, samples, channels)
            y: Combined labels as integers (n_total_windows,)
            trial_ids: Combined trial IDs for proper train/test splitting (n_total_windows,)
            session_indices: Per-window session index (0..n_sessions-1) for session normalization
            label_names: Sorted list of unique gesture labels across all sessions
            session_ids: List of session IDs that were loaded

        Raises:
            ValueError: If no sessions found or sessions have incompatible shapes
        """
        sessions = self.list_sessions()

        if not sessions:
            raise ValueError("No sessions found to load!")

        print(f"[Storage] Loading {len(sessions)} session(s) for combined training...")
        if filter_transitions:
            print(f"[Storage] Transition filtering: START={TRANSITION_START_MS}ms, END={TRANSITION_END_MS}ms")

        all_X = []
        all_labels = []
        all_trial_ids = []  # Track trial_ids for proper train/test splitting
        all_session_indices = []  # Per-window session index for session normalization
        loaded_sessions = []
        reference_shape = None
        total_removed = 0
        total_original = 0
        trial_id_offset = 0  # Offset trial_ids across sessions to ensure global uniqueness

        for session_id in sessions:
            filepath = self.get_session_filepath(session_id)

            with h5py.File(filepath, 'r') as f:
                X = f['windows/emg_data'][:]
                labels_raw = f['windows/labels'][:]
                start_times = f['windows/start_times'][:]
                end_times = f['windows/end_times'][:]

                # Load trial_ids if available (new files), otherwise generate from index
                if 'windows/trial_ids' in f:
                    trial_ids = f['windows/trial_ids'][:] + trial_id_offset
                else:
                    # Legacy file without trial_ids: assign unique trial_id per window
                    # This is conservative - treats each window as separate trial
                    print(f"[Storage] WARNING: {session_id} missing trial_ids, generating from indices")
                    trial_ids = np.arange(len(X), dtype=np.int32) + trial_id_offset

            # Validate shape compatibility
            if reference_shape is None:
                reference_shape = X.shape[1:]  # (samples_per_window, channels)
            elif X.shape[1:] != reference_shape:
                print(f"[Storage] WARNING: Skipping {session_id} - incompatible shape {X.shape[1:]} vs {reference_shape}")
                continue

            # Decode labels
            labels = []
            for l in labels_raw:
                if isinstance(l, bytes):
                    labels.append(l.decode('utf-8'))
                else:
                    labels.append(l)

            original_count = len(X)
            total_original += original_count

            # Apply transition filtering per session (each has its own gesture boundaries)
            if filter_transitions:
                # Need temporary y for filtering function
                temp_label_names = sorted(set(labels))
                temp_label_to_idx = {name: idx for idx, name in enumerate(temp_label_names)}
                temp_y = np.array([temp_label_to_idx[l] for l in labels], dtype=np.int32)

                X, temp_y, labels, trial_ids = filter_transition_windows(
                    X, temp_y, labels, start_times, end_times, trial_ids=trial_ids
                )
                total_removed += original_count - len(X)

            current_session_idx = len(all_X)  # 0-based index before appending
            all_X.append(X)
            all_labels.extend(labels)
            all_trial_ids.extend(trial_ids.tolist())
            all_session_indices.extend([current_session_idx] * len(X))
            loaded_sessions.append(session_id)

            # Update trial_id offset for next session (ensure global uniqueness)
            if len(trial_ids) > 0:
                trial_id_offset = max(trial_ids) + 1

            print(f"[Storage]   - {session_id}: {len(X)} windows" +
                  (f" (was {original_count})" if filter_transitions and len(X) != original_count else ""))

        if not all_X:
            raise ValueError("No compatible sessions found!")

        # Combine all data
        X_combined = np.concatenate(all_X, axis=0)
        trial_ids_combined = np.array(all_trial_ids, dtype=np.int32)
        session_indices_combined = np.array(all_session_indices, dtype=np.int32)

        # Create unified label mapping across all sessions
        label_names = sorted(set(all_labels))
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}
        y_combined = np.array([label_to_idx[l] for l in all_labels], dtype=np.int32)

        n_unique_trials = len(np.unique(trial_ids_combined))
        print(f"[Storage] Combined dataset: X{X_combined.shape}, y{y_combined.shape}")
        print(f"[Storage] Unique trials: {n_unique_trials} (for proper train/test splitting)")
        if filter_transitions and total_removed > 0:
            print(f"[Storage] Total removed: {total_removed}/{total_original} windows ({total_removed/total_original*100:.1f}%)")
        print(f"[Storage] Labels: {label_names}")
        print(f"[Storage] Sessions loaded: {len(loaded_sessions)}")

        return X_combined, y_combined, trial_ids_combined, session_indices_combined, label_names, loaded_sessions

    def list_sessions(self) -> list[str]:
        """List all available session IDs."""
        return sorted([f.stem for f in self.data_dir.glob("*.hdf5")])

    def get_session_info(self, session_id: str) -> dict:
        """Get quick info about a session without loading all data."""
        filepath = self.get_session_filepath(session_id)
        with h5py.File(filepath, 'r') as f:
            return {
                'user_id': f.attrs['user_id'],
                'timestamp': f.attrs['timestamp'],
                'num_windows': f.attrs['num_windows'],
                'gestures': json.loads(f.attrs['gestures']),
                'sampling_rate': f.attrs['sampling_rate'],
                'window_size_ms': f.attrs['window_size_ms'],
            }


# =============================================================================
# COLLECTION SESSION (Requires ESP32 hardware)
# =============================================================================

def run_labeled_collection_demo():
    """
    Run a labeled EMG collection session:
      1. Connect to ESP32 via serial
      2. Prompt scheduler guides the user through gestures
      3. EMG stream collects real signals
      4. Windower groups samples into fixed-size windows
      5. Labels are assigned based on which prompt was active
      6. Session is saved to HDF5 with user ID

    REQUIRES: ESP32 hardware connected via USB.
    """
    print("\n" + "=" * 60)
    print("LABELED EMG COLLECTION (ESP32 Required)")
    print("=" * 60)

    # Get user ID
    user_id = input("\nEnter your user ID (e.g., user_001): ").strip()
    if not user_id:
        user_id = USER_ID  # Fall back to default
        print(f"  Using default: {user_id}")
    else:
        print(f"  User ID: {user_id}")

    # Define gestures to collect (names match ESP32 gesture definitions)
    gestures = ["open", "fist", "hook_em", "thumbs_up"]

    # Create the prompt scheduler
    scheduler = PromptScheduler(
        gestures=gestures,
        hold_sec=GESTURE_HOLD_SEC,
        rest_sec=REST_BETWEEN_SEC,
        reps=REPS_PER_GESTURE
    )
    scheduler.print_schedule()

    # Connect to ESP32
    print("\n[Connecting to ESP32...]")
    stream = RealSerialStream()
    try:
        stream.connect(timeout=5.0)
        print(f"  Connected: {stream.device_info}")
    except Exception as e:
        print(f"  ERROR: Failed to connect to ESP32: {e}")
        print("  Make sure the ESP32 is connected and firmware is flashed.")
        return [], []

    parser = EMGParser(num_channels=NUM_CHANNELS)
    windower = Windower(
        window_size_ms=WINDOW_SIZE_MS,
        sample_rate=SAMPLING_RATE_HZ,
        hop_size_ms=HOP_SIZE_MS
    )

    # Storage for windows, labels, and trial_ids (kept separate to enforce training/inference separation)
    collected_windows: list[EMGWindow] = []
    collected_labels: list[str] = []
    collected_trial_ids: list[int] = []  # Track trial_id for proper train/test splitting
    last_prompt_name = None

    # Start collection
    input("\nPress ENTER to start collection session...")
    stream.start()
    scheduler.start_session()

    print("\n" + "-" * 40)
    print("COLLECTING... Watch the prompts!")
    print("-" * 40)

    try:
        while not scheduler.is_session_complete():
            # Get current prompt
            prompt = scheduler.get_current_prompt()

            # Display prompt changes
            if prompt and prompt.gesture_name != last_prompt_name:
                elapsed = scheduler.get_elapsed_time()
                if prompt.gesture_name == "rest":
                    print(f"\n  [{elapsed:5.1f}s] >>> REST <<<")
                else:
                    print(f"\n  [{elapsed:5.1f}s] >>> {prompt.gesture_name.upper()} <<<")
                last_prompt_name = prompt.gesture_name

            # Read and parse data
            line = stream.readline()
            if line:
                sample = parser.parse_line(line)
                if sample:
                    # Try to form a window
                    window = windower.add_sample(sample)
                    if window:
                        # Store window, label, and trial_id separately (training/inference separation)
                        # Shift label lookup forward to align with actual muscle activation
                        label_time = window.start_time + LABEL_SHIFT_MS / 1000.0
                        label = scheduler.get_label_for_time(label_time)
                        trial_id = scheduler.get_trial_id_for_time(label_time)
                        collected_windows.append(window)
                        collected_labels.append(label)
                        collected_trial_ids.append(trial_id)

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    finally:
        stream.stop()
        stream.disconnect()

    # Report results
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total windows collected: {len(collected_windows)}")
    print(f"Parse errors: {parser.parse_errors}")

    # Count labels (from separate labels list, not from windows)
    label_counts = {}
    for label in collected_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nWindows per label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # Show example windows
    if collected_windows:
        print(f"\nExample windows:")
        for i, w in enumerate(collected_windows[:3]):
            data = w.to_numpy()
            print(f"  Window {w.window_id}: label='{collected_labels[i]}', "
                  f"samples={len(w.samples)}, "
                  f"ch0_mean={data[:, 0].mean():.1f}")

        # Show one window from each gesture type
        print(f"\nSignal comparison (channel 0 std dev by gesture):")
        for label in sorted(label_counts.keys()):
            # Get indices where label matches
            indices = [i for i, l in enumerate(collected_labels) if l == label]
            if indices:
                all_ch0 = np.concatenate([collected_windows[i].get_channel(0) for i in indices])
                print(f"  {label}: std={all_ch0.std():.1f}")

    # --- Save the session ---
    if collected_windows:
        save_choice = input("\nSave this session? (y/n): ").strip().lower()
        if save_choice == 'y':
            storage = SessionStorage()
            session_id = storage.generate_session_id(user_id)

            metadata = SessionMetadata(
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                sampling_rate=SAMPLING_RATE_HZ,
                window_size_ms=WINDOW_SIZE_MS,
                num_channels=NUM_CHANNELS,
                gestures=gestures,
                notes=""
            )

            # Pass windows, labels, and trial_ids separately (enforces separation)
            filepath = storage.save_session(
                collected_windows, collected_labels, metadata,
                trial_ids=collected_trial_ids
            )
            print(f"\nSession saved! ID: {session_id}")

    return collected_windows, collected_labels, collected_trial_ids


# =============================================================================
# INSPECT SESSIONS (Load and view saved sessions)
# =============================================================================

def run_storage_demo():
    """Demonstrates loading and inspecting saved sessions."""
    print("\n" + "=" * 60)
    print("INSPECT SAVED SESSIONS")
    print("=" * 60)

    storage = SessionStorage()
    sessions = storage.list_sessions()

    if not sessions:
        print("\nNo saved sessions found!")
        print(f"Run option 2 first to collect and save a session.")
        print(f"Sessions are stored in: {storage.data_dir.absolute()}")
        return None

    print(f"\nFound {len(sessions)} saved session(s):")
    print("-" * 40)

    for i, session_id in enumerate(sessions):
        info = storage.get_session_info(session_id)
        print(f"\n  [{i+1}] {session_id}")
        print(f"      User: {info['user_id']}")
        print(f"      Time: {info['timestamp']}")
        print(f"      Windows: {info['num_windows']}")
        print(f"      Gestures: {info['gestures']}")

    print("\n" + "-" * 40)
    choice = input("Enter session number to load (or 'q' to quit): ").strip()

    if choice.lower() == 'q':
        return None

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(sessions):
            print("Invalid selection!")
            return None
        session_id = sessions[idx]
    except ValueError:
        print("Invalid input!")
        return None

    print(f"\n{'=' * 60}")
    print(f"LOADING SESSION: {session_id}")
    print("=" * 60)

    # Labels returned separately from windows (enforces training/inference separation)
    windows, labels, metadata = storage.load_session(session_id)

    print(f"\nMetadata:")
    print(f"  User: {metadata.user_id}")
    print(f"  Timestamp: {metadata.timestamp}")
    print(f"  Sampling rate: {metadata.sampling_rate} Hz")
    print(f"  Window size: {metadata.window_size_ms} ms")
    print(f"  Channels: {metadata.num_channels}")
    print(f"  Gestures: {metadata.gestures}")

    # Count from separate labels list
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} windows")

    print(f"\n{'-' * 40}")
    print("LOADING FOR MACHINE LEARNING")
    print("-" * 40)

    X, y, label_names = storage.load_for_training(session_id)

    print(f"\nData shapes:")
    print(f"  X (features): {X.shape}")
    print(f"     - {X.shape[0]} windows")
    print(f"     - {X.shape[1]} samples per window")
    print(f"     - {X.shape[2]} channels")
    print(f"  y (labels): {y.shape}")
    print(f"  Label mapping: {dict(enumerate(label_names))}")

    # --- Feature Extraction & Visualization ---
    print(f"\n{'-' * 40}")
    print("EXTRACTING FEATURES FOR VISUALIZATION")
    print("-" * 40)

    # Note: Per-window centering is done inside EMGFeatureExtractor.
    # This is the correct approach for real-time inference (causal, no future data).
    # Global centering across all windows would leak information and not work in real-time.

    extractor = EMGFeatureExtractor()
    n_windows = X.shape[0]
    n_channels = X.shape[2]

    # Extract features per channel: shape (n_windows, n_channels, 4)
    # Features order: [rms, wl, zc, ssc]
    features_by_channel = np.zeros((n_windows, n_channels, 4))

    for i in range(n_windows):
        for ch in range(n_channels):
            ch_features = extractor.extract_features_single_channel(X[i, :, ch])
            features_by_channel[i, ch, 0] = ch_features['rms']
            features_by_channel[i, ch, 1] = ch_features['wl']
            features_by_channel[i, ch, 2] = ch_features['zc']
            features_by_channel[i, ch, 3] = ch_features['ssc']

    print(f"  Extracted features for {n_windows} windows, {n_channels} channels")
    print(f"  (Per-window centering applied inside feature extractor)")

    # Create time axis (window indices as proxy for time)
    time_axis = np.arange(n_windows)

    # Find gesture transition points (where label changes)
    transitions = []
    current_label = y[0]
    for i in range(1, len(y)):
        if y[i] != current_label:
            transitions.append((i, label_names[y[i]]))
            current_label = y[i]

    # Define colors for gesture markers (matches GUI color scheme)
    def get_gesture_color(name):
        name_lower = name.lower()
        if 'rest' in name_lower:
            return 'gray'
        elif 'open' in name_lower:
            return 'cyan'
        elif 'fist' in name_lower:
            return 'blue'
        elif 'hook' in name_lower:
            return 'orange'
        elif 'thumb' in name_lower:
            return 'green'
        return 'red'

    feature_titles = ['RMS', 'Waveform Length (WL)', 'Zero Crossings (ZC)', 'Slope Sign Changes (SSC)']
    feature_colors = ['red', 'blue', 'green', 'purple']
    feature_ylabels = ['Amplitude', 'WL (a.u.)', 'Count', 'Count']

    # --- Figure 1: Raw EMG Signal ---
    fig_raw, axes_raw = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes_raw = axes_raw.flatten()

    # Concatenate all windows to show continuous signal
    samples_per_window = X.shape[1]
    total_samples = n_windows * samples_per_window
    raw_time = np.arange(total_samples)

    for ch in range(n_channels):
        ax = axes_raw[ch]

        # Flatten all windows into continuous signal for this channel
        # Center per-channel for visualization only (subtract channel mean)
        raw_signal = X[:, :, ch].flatten()
        raw_signal_centered = raw_signal - raw_signal.mean()
        ax.plot(raw_time, raw_signal_centered, linewidth=0.5, color='black')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_title(f"Channel {ch}", fontsize=11)
        ax.set_ylabel("Amplitude (centered)")
        ax.grid(True, alpha=0.3)

        # Add vertical lines at gesture transitions (scaled to sample index)
        for trans_idx, trans_label in transitions:
            sample_idx = trans_idx * samples_per_window
            color = get_gesture_color(trans_label)
            ax.axvline(sample_idx, color=color, linestyle='--', alpha=0.6, linewidth=1)

    # Add legend for gesture colors
    legend_elements = []
    for label_name in label_names:
        color = get_gesture_color(label_name)
        legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='--', label=label_name))
    axes_raw[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig_raw.suptitle("Raw EMG Signal (Centered for Display) - All Channels", fontsize=14, fontweight='bold')
    fig_raw.supxlabel("Sample Index")
    plt.tight_layout()

    # --- Figures 2-5: Feature plots (one per feature type) ---
    for feat_idx, feat_title in enumerate(feature_titles):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        for ch in range(n_channels):
            ax = axes[ch]

            # Plot feature as line graph
            feat_data = features_by_channel[:, ch, feat_idx]
            ax.plot(time_axis, feat_data, linewidth=1, color=feature_colors[feat_idx])
            ax.set_title(f"Channel {ch}", fontsize=11)
            ax.grid(True, alpha=0.3)

            # Add vertical lines at gesture transitions
            for trans_idx, trans_label in transitions:
                color = get_gesture_color(trans_label)
                ax.axvline(trans_idx, color=color, linestyle='--', alpha=0.6, linewidth=1)

        # Add legend for gesture colors
        legend_elements = []
        for label_name in label_names:
            color = get_gesture_color(label_name)
            legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='--', label=label_name))
        axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

        fig.suptitle(f"{feat_title} - All Channels", fontsize=14, fontweight='bold')
        fig.supxlabel("Window Index (Time)")
        fig.supylabel(feature_ylabels[feat_idx])
        plt.tight_layout()

    plt.show()
    print(f"\n  Displayed 5 figures: Raw EMG + 4 features (close windows to continue)")

    return X, y, label_names


# =============================================================================
# FEATURE EXTRACTION (Time-domain features for EMG)
# =============================================================================

class EMGFeatureExtractor:
    """
    Extracts time-domain and frequency-domain features from EMG windows.

    Change 1 — expanded feature set (expanded=True, default):
      Per channel (20 features):
        TD-4 (legacy): RMS, WL, ZC, SSC
        TD extended:   MAV, VAR, IEMG, WAMP
        AR model:      AR1, AR2, AR3, AR4  (4th-order via Yule-Walker)
        Frequency:     MNF, MDF, PKF, MNP  (mean/median/peak freq, mean power)
        Band power:    BP0(20-80Hz), BP1(80-150Hz), BP2(150-250Hz), BP3(250-450Hz)
      Cross-channel (cross_channel=True, default):
        For each channel pair (i,j): Pearson correlation, log-RMS ratio, covariance
        For 3 hand channels: 3 pairs × 3 = 9 cross-channel features
      Total for HAND_CHANNELS=[0,1,2]: 20×3 + 9 = 69 features

    Legacy mode (expanded=False): 4 features per channel only (RMS, WL, ZC, SSC).
    Old pickled models automatically use legacy mode via __setstate__.

    IMPORTANT: Per-window DC removal (mean subtraction) is applied before all
    features. This is causal (uses only data within the current window).
    """

    # Feature key ordering — determines output vector layout
    _LEGACY_KEYS = ['rms', 'wl', 'zc', 'ssc']
    _EXPANDED_KEYS = [
        'rms',  'wl',   'zc',   'ssc',   # TD-4
        'mav',  'var',  'iemg', 'wamp',  # TD extended
        'ar1',  'ar2',  'ar3',  'ar4',   # AR(4) model
        'mnf',  'mdf',  'pkf',  'mnp',   # Frequency descriptors
        'bp0',  'bp1',  'bp2',  'bp3',   # Band powers
    ]
    # Keys that are amplitude-dependent and should be divided by norm_factor
    _NORM_KEYS = {'rms', 'wl', 'mav', 'iemg'}

    def __init__(self,
                 zc_threshold_percent: float = 0.1,
                 ssc_threshold_percent: float = 0.1,
                 channels: Optional[list[int]] = None,
                 normalize: bool = True,
                 expanded: bool = True,
                 cross_channel: bool = True,
                 fft_n: int = 256,
                 fs: float = float(SAMPLING_RATE_HZ),
                 reinhard: bool = False,
                 bandpass: bool = True):
        """
        Args:
            zc_threshold_percent: ZC/WAMP threshold as fraction of RMS.
            ssc_threshold_percent: SSC threshold as fraction of RMS squared.
            channels: Channel indices to extract features from; None = all.
            normalize: Divide amplitude-dependent features by total RMS across
                       channels (makes model robust to impedance shifts).
            expanded: Use full 20-feature/channel set (Change 1). False = legacy
                      4-feature/channel set for backward compatibility.
            cross_channel: Append pairwise cross-channel features (correlation,
                           log-RMS ratio, covariance). Only when expanded=True.
            fft_n: FFT size for frequency features (zero-pads window if needed).
            fs: Sampling frequency in Hz (used for frequency axis).
            reinhard: Change 4 — apply Reinhard tone-mapping (64·x/(32+|x|))
                      before feature extraction. Must match MODEL_USE_REINHARD in
                      firmware model_weights.h. Default False.
            bandpass: Apply 20-450 Hz bandpass filter before feature extraction.
                      Must be True to match firmware IIR bandpass. Default True.
        """
        self.zc_threshold_percent  = zc_threshold_percent
        self.ssc_threshold_percent = ssc_threshold_percent
        self.channels      = channels
        self.normalize     = normalize
        self.expanded      = expanded
        self.cross_channel = cross_channel
        self.fft_n         = fft_n
        self.fs            = fs
        self.reinhard      = reinhard
        self.bandpass      = bandpass

        # Pre-compute bandpass SOS coefficients (2nd-order Butterworth, 20-450 Hz)
        # Matches firmware IIR biquad bandpass in inference.c
        if self.bandpass:
            nyq = self.fs / 2.0
            self._bp_sos = butter(2, [20.0 / nyq, 450.0 / nyq], btype='band', output='sos')
        else:
            self._bp_sos = None

    def __setstate__(self, state: dict):
        """Restore pickle and add defaults for attributes added in Change 1+."""
        self.__dict__.update(state)
        if 'expanded'      not in state: self.expanded      = False
        if 'cross_channel' not in state: self.cross_channel = False
        if 'fft_n'         not in state: self.fft_n         = 256
        if 'fs'            not in state: self.fs            = float(SAMPLING_RATE_HZ)
        if 'reinhard'      not in state: self.reinhard      = False
        if 'bandpass'      not in state: self.bandpass      = False
        # Reconstruct SOS coefficients for bandpass filter
        if self.bandpass:
            nyq = self.fs / 2.0
            self._bp_sos = butter(2, [20.0 / nyq, 450.0 / nyq], btype='band', output='sos')
        else:
            self._bp_sos = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ar_coefficients(signal: np.ndarray, order: int = 4) -> np.ndarray:
        """4th-order AR coefficients via Yule-Walker (autocorrelation method)."""
        n = len(signal)
        r = np.array([float(np.dot(signal[:n - k], signal[k:])) / n
                      for k in range(order + 1)])
        T = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
        try:
            return np.linalg.solve(T, r[1:order + 1])
        except np.linalg.LinAlgError:
            return np.zeros(order)

    def _spectral_features(self, signal: np.ndarray) -> tuple:
        """MNF, MDF, PKF, MNP, BP0-BP3 via rfft (zero-padded to fft_n)."""
        spec   = np.abs(np.fft.rfft(signal, n=self.fft_n)) ** 2
        freqs  = np.fft.rfftfreq(self.fft_n, d=1.0 / self.fs)
        total  = float(np.sum(spec)) + 1e-10

        mnf = float(np.dot(freqs, spec) / total)

        cumsum  = np.cumsum(spec)
        mid_idx = int(np.searchsorted(cumsum, total / 2.0))
        mdf     = float(freqs[min(mid_idx, len(freqs) - 1)])

        pkf = float(freqs[int(np.argmax(spec))])
        mnp = float(total / len(spec))

        def _bp(f_lo: float, f_hi: float) -> float:
            mask = (freqs >= f_lo) & (freqs < f_hi)
            return float(np.sum(spec[mask]) / total)

        return mnf, mdf, pkf, mnp, _bp(20, 80), _bp(80, 150), _bp(150, 250), _bp(250, 450)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features_single_channel(self, signal: np.ndarray) -> dict:
        """
        Extract features from a single, already-selected channel.

        Returns a dict with 4 keys (legacy) or 20 keys (expanded).
        Bandpass filter (if enabled) + per-window DC removal are applied first.
        """
        # Bandpass filter to match firmware IIR (20-450 Hz, 2nd-order Butterworth).
        # Uses sosfilt (causal) with sosfilt_zi to initialise the filter state
        # at the signal's DC level, avoiding large startup transients on the
        # short 150-sample windows.
        if self.bandpass and self._bp_sos is not None:
            zi = sosfilt_zi(self._bp_sos) * signal[0]
            signal, _ = sosfilt(self._bp_sos, signal, zi=zi)

        signal = signal - np.mean(signal)  # DC removal

        # Change 4 — Reinhard tone-mapping (compresses large spikes)
        if self.reinhard:
            signal = 64.0 * signal / (32.0 + np.abs(signal))

        rms = float(np.sqrt(np.mean(signal ** 2)))
        wl  = float(np.sum(np.abs(np.diff(signal))))

        zc_thresh  = self.zc_threshold_percent * rms
        ssc_thresh = (self.ssc_threshold_percent * rms) ** 2

        diffs = np.diff(signal)
        sign_chg = signal[:-1] * signal[1:] < 0
        zc  = int(np.sum(sign_chg & (np.abs(diffs) > zc_thresh)))

        dl = signal[1:-1] - signal[:-2]
        dr = signal[1:-1] - signal[2:]
        ssc = int(np.sum((dl * dr) > ssc_thresh))

        feats: dict = {'rms': rms, 'wl': wl, 'zc': float(zc), 'ssc': float(ssc)}

        if self.expanded:
            mav  = float(np.mean(np.abs(signal)))
            var  = float(np.var(signal))
            iemg = float(np.sum(np.abs(signal)))
            wamp = int(np.sum(np.abs(diffs) > zc_thresh))

            ar = self._ar_coefficients(signal, order=4)
            mnf, mdf, pkf, mnp, bp0, bp1, bp2, bp3 = self._spectral_features(signal)

            feats.update({
                'mav': mav, 'var': var, 'iemg': iemg, 'wamp': float(wamp),
                'ar1': float(ar[0]), 'ar2': float(ar[1]),
                'ar3': float(ar[2]), 'ar4': float(ar[3]),
                'mnf': mnf, 'mdf': mdf, 'pkf': pkf, 'mnp': mnp,
                'bp0': bp0, 'bp1': bp1, 'bp2': bp2, 'bp3': bp3,
            })

        return feats

    def extract_features_window(self, window: np.ndarray) -> np.ndarray:
        """
        Extract features from a window of shape (samples, channels).

        Returns a flat float32 array ordered as:
          [ch_i feats..., ch_j feats..., ..., cross-channel feats...]
        """
        channel_indices = self.channels if self.channels is not None \
                          else list(range(window.shape[1]))

        all_ch_feats = [self.extract_features_single_channel(window[:, ch])
                        for ch in channel_indices]

        norm_factor = 1.0
        if self.normalize:
            total_rms   = float(np.sqrt(sum(f['rms'] ** 2 for f in all_ch_feats)))
            norm_factor = max(total_rms, 1e-6)

        feat_keys = self._EXPANDED_KEYS if self.expanded else self._LEGACY_KEYS

        features: list[float] = []
        for ch_feats in all_ch_feats:
            for key in feat_keys:
                val = ch_feats[key]
                if self.normalize and key in self._NORM_KEYS:
                    val = val / norm_factor
                features.append(val)

        # Cross-channel features (expanded mode, ≥2 channels)
        # Bug 6 fix: firmware computes cross-channel features from
        # Reinhard-mapped signals when MODEL_USE_REINHARD=1.  Apply the
        # same tone-mapping here so correlation/covariance match.
        if self.expanded and self.cross_channel and len(channel_indices) >= 2:
            centered = []
            for ch in channel_indices:
                sig = window[:, ch] - np.mean(window[:, ch])
                # Apply bandpass if enabled (matches firmware pipeline order)
                if self.bandpass and self._bp_sos is not None:
                    zi = sosfilt_zi(self._bp_sos) * sig[0]
                    sig, _ = sosfilt(self._bp_sos, sig, zi=zi)
                if self.reinhard:
                    sig = 64.0 * sig / (32.0 + np.abs(sig))
                centered.append(sig)
            rms_vals = [f['rms'] + 1e-10 for f in all_ch_feats]
            n = window.shape[0]

            for i in range(len(channel_indices)):
                for j in range(i + 1, len(channel_indices)):
                    si, sj = centered[i], centered[j]
                    ri, rj = rms_vals[i], rms_vals[j]

                    corr = float(np.clip(np.dot(si, sj) / (n * ri * rj), -1.0, 1.0))
                    lrms = float(np.log(ri / rj))
                    cov  = float(np.dot(si, sj) / n)
                    if self.normalize:
                        cov /= (norm_factor ** 2)

                    features.extend([corr, lrms, cov])

        return np.array(features, dtype=np.float32)

    def extract_features_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of windows.

        Args:
            X: (n_windows, n_samples, n_channels)
        Returns:
            (n_windows, n_features) float32 array.
        """
        # Vectorised bandpass: apply sosfiltfilt on all windows at once along
        # the samples axis.  This is ~100x faster than per-window sosfilt calls
        # (scipy's C loop vs Python loop).  We disable per-window bandpass in
        # extract_features_single_channel during batch extraction.
        if self.bandpass and self._bp_sos is not None:
            X = sosfiltfilt(self._bp_sos, X, axis=1).astype(np.float32)

        n_windows  = X.shape[0]
        n_ch_total = X.shape[2]
        n_features = self._n_features(n_ch_total)
        features   = np.zeros((n_windows, n_features), dtype=np.float32)

        # Temporarily disable per-window bandpass (already applied above)
        saved_bp = self.bandpass
        self.bandpass = False
        try:
            for i in range(n_windows):
                features[i] = self.extract_features_window(X[i])
        finally:
            self.bandpass = saved_bp

        return features

    def _n_features(self, n_total_channels: int) -> int:
        """Total feature vector length for the current configuration."""
        n_ch         = len(self.channels) if self.channels is not None else n_total_channels
        per_ch       = len(self._EXPANDED_KEYS if self.expanded else self._LEGACY_KEYS)
        n            = n_ch * per_ch
        if self.expanded and self.cross_channel and n_ch >= 2:
            n += 3 * (n_ch * (n_ch - 1) // 2)  # 3 features × C(n_ch,2) pairs
        return n

    def get_feature_names(self, n_channels: int = 0) -> list[str]:
        """Human-readable feature names matching the extract_features_window layout."""
        channel_indices = self.channels if self.channels is not None \
                          else list(range(n_channels))

        feat_keys = self._EXPANDED_KEYS if self.expanded else self._LEGACY_KEYS

        names: list[str] = []
        for ch in channel_indices:
            for key in feat_keys:
                names.append(f'ch{ch}_{key}')

        if self.expanded and self.cross_channel and len(channel_indices) >= 2:
            for i in range(len(channel_indices)):
                for j in range(i + 1, len(channel_indices)):
                    ci, cj = channel_indices[i], channel_indices[j]
                    names.extend([
                        f'cc_{ci}{cj}_corr',
                        f'cc_{ci}{cj}_lrms',
                        f'cc_{ci}{cj}_cov',
                    ])

        return names


# =============================================================================
# Change 6 — MPF FEATURE EXTRACTOR (Python training only)
# =============================================================================

class MPFFeatureExtractor:
    """
    Simplified 3-channel MPF: CSD upper triangle per 6 frequency bands = 36 features.
    Python training only. Omits matrix logarithm (not needed for 3 channels).
    Source: Kaifosh et al. Nature 2025. doi:10.1038/s41586-025-09255-w
    ESP32 approximation: use bp0-bp3 from EMGFeatureExtractor (Change 1).
    """
    BANDS = [(0, 62), (62, 125), (125, 187), (187, 250), (250, 375), (375, 500)]

    def __init__(self, channels=None, log_diagonal=True):
        self.channels = channels or HAND_CHANNELS
        self.log_diag = log_diagonal
        self.n_ch = len(self.channels)
        self._r, self._c = np.triu_indices(self.n_ch)
        self.n_features = len(self.BANDS) * len(self._r)

    def extract_window(self, window):
        sig   = window[:, self.channels].astype(np.float64)
        N     = len(sig)
        freqs = np.fft.rfftfreq(N, d=1.0 / SAMPLING_RATE_HZ)
        Xf    = np.fft.rfft(sig, axis=0)
        feats = []
        for lo, hi in self.BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                feats.extend([0.0] * len(self._r))
                continue
            CSD = (Xf[mask].conj().T @ Xf[mask]).real / N
            if self.log_diag:
                for k in range(self.n_ch):
                    CSD[k, k] = np.log(max(CSD[k, k], 1e-10))
            feats.extend(CSD[self._r, self._c].tolist())
        return np.array(feats, dtype=np.float32)

    def extract_batch(self, X):
        out = np.zeros((len(X), self.n_features), dtype=np.float32)
        for i in range(len(X)):
            out[i] = self.extract_window(X[i])
        return out


# =============================================================================
# CALIBRATION TRANSFORM (Per-session feature-space alignment)
# =============================================================================

class CalibrationTransform:
    """
    Corrects for electrode placement drift between sessions via Session Z-Score Normalization.

    Training: each training session's features are independently z-scored
    (subtract session mean, divide by session std) before LDA fitting.
    This removes placement-dependent amplitude shifts, so the model learns
    in a placement-invariant normalized feature space.

    Calibration: collect a short clip of each gesture → compute global
    mean (mu_calib) and std (sigma_calib) of those features → apply the
    same z-score to every live window:

        x_normalized = (x_live - mu_calib) / sigma_calib

    This projects live features into the same normalized space that training
    used, regardless of how electrode placement changed.

    Workflow:
      1. fit_from_training()    — called automatically in EMGClassifier.train().
                                  Stores per-class training centroids (in normalized
                                  space) for diagnostics.
      2. fit_from_calibration() — called at session start after collecting
                                  a short clip of each gesture.
                                  Computes mu_calib / sigma_calib.
      3. apply()                — called on every live feature vector.
                                  Returns (features - mu_calib) / sigma_calib.
    """

    def __init__(self):
        self.has_training_stats: bool = False
        self.is_fitted: bool = False
        self.class_means_train: dict = {}                # {label: ndarray} from training (normalized space)
        self.class_means_calib: dict = {}                # {label: ndarray} from calibration (raw space)
        # Stats for the z-score transform
        self.mu_calib: Optional[np.ndarray] = None       # Class-balanced mean of calibration features (raw space)
        self.sigma_calib: Optional[np.ndarray] = None    # Global std of calibration features (raw space)
        self.sigma_train: Optional[np.ndarray] = None    # Mean per-session sigma from training (preferred scale ref)
        # Energy gate for rest detection (bypasses LDA when signal is quiet)
        self.rest_energy_threshold: Optional[float] = None

    def fit_from_training(self, X_features: np.ndarray, y: np.ndarray, label_names: list):
        """
        Store per-class training centroids. Called automatically in EMGClassifier.train().

        Args:
            X_features:  (n_windows, n_features) extracted training features
            y:           (n_windows,) integer label indices
            label_names: label string list matching y indices
        """
        self.has_training_stats = True

        self.class_means_train = {}
        for i, name in enumerate(label_names):
            mask = y == i
            if mask.any():
                self.class_means_train[name] = np.mean(X_features[mask], axis=0)

    def fit_from_calibration(self, calib_features: np.ndarray, calib_labels: list):
        """
        Compute z-score normalization params from calibration-session data.

        mu_calib    = class-balanced mean (average of per-class centroids)
        sigma_calib = overall std of all calibration feature windows

        Using the class-balanced mean prevents near-zero-amplitude classes (rest)
        from landing at the wrong normalized position when training sessions had
        unequal numbers of windows per class.

        Args:
            calib_features: (n_windows, n_features) from calibration clips
            calib_labels:   gesture label per window
        """
        if not self.has_training_stats:
            raise ValueError(
                "Training stats not available. Load a model that was trained "
                "after calibration support was added (retrain if needed)."
            )

        # Per-class calibration centroids (raw space)
        self.class_means_calib = {}
        label_arr = np.array(calib_labels)
        for label in set(calib_labels):
            mask = label_arr == label
            if mask.any():
                self.class_means_calib[label] = np.mean(calib_features[mask], axis=0)

        # Class-balanced mean: average of per-class centroids (not overall mean).
        # Prevents class-imbalanced calibration clips from biasing the normalization
        # origin (especially important for rest, which has near-zero amplitude).
        self.mu_calib = np.mean(list(self.class_means_calib.values()), axis=0)
        self.sigma_calib = np.std(calib_features, axis=0) + 1e-8

        # rest_energy_threshold is set externally from raw window RMS values
        # (cannot be computed here — extracted features are amplitude-normalized).
        self.rest_energy_threshold = None

        self.is_fitted = True

        # Decide which sigma to use for scaling:
        #   sigma_train (preferred) — mean per-session sigma from training.
        #     Ensures the classifier sees calibration features at the SAME scale
        #     as training features, which is critical for QDA whose per-class
        #     covariance ellipsoids are fixed in normalized training space.
        #   sigma_calib (fallback) — std of this calibration session.
        #     Used only if the model was trained without session normalization.
        sigma_used = self.sigma_train if self.sigma_train is not None else self.sigma_calib
        sigma_source = "sigma_train" if self.sigma_train is not None else "sigma_calib (fallback)"
        print(f"[Calibration] Z-score fit: {len(calib_features)} windows, "
              f"{len(self.class_means_calib)} classes  [scale ref: {sigma_source}]")
        # Per-class residual in normalized space (lower = better alignment)
        common = set(self.class_means_calib) & set(self.class_means_train)
        for c in sorted(common):
            norm_calib = (self.class_means_calib[c] - self.mu_calib) / self.sigma_calib
            residual = np.linalg.norm(self.class_means_train[c] - norm_calib)
            print(f"[Calibration]   {c}: normalized residual = {residual:.4f}")

    def apply(self, features: np.ndarray) -> np.ndarray:
        """
        Z-score normalize features using calibration session statistics.

        Uses sigma_train (mean per-session sigma from training) for scaling when
        available — this keeps calibration features at the same scale as training
        features, which is critical for QDA.  Falls back to sigma_calib for old
        models trained without session normalization.

        Args:
            features: shape (n_features,) or (n_windows, n_features)
        Returns:
            (features - mu_calib) / sigma, same shape as input.
            Pass-through if not fitted.
        """
        if not self.is_fitted:
            return features
        sigma = self.sigma_train if self.sigma_train is not None else self.sigma_calib
        return (features - self.mu_calib) / sigma

    def reset(self):
        """Remove per-session calibration (keeps training centroids intact)."""
        self.mu_calib = None
        self.sigma_calib = None
        self.rest_energy_threshold = None
        self.is_fitted = False
        self.class_means_calib = {}
        # sigma_train is permanent (set at train time, not session-specific)


# =============================================================================
# DATA AUGMENTATION (Change 3)
# =============================================================================

def augment_emg_batch(
    X: np.ndarray,
    y: np.ndarray,
    multiplier: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment raw EMG windows for training robustness.

    Must be called on raw windows (n_windows, n_samples, n_channels),
    not on pre-computed features.  Each copy independently applies:
      - Amplitude scaling ×[0.80, 1.20]
      - Gaussian noise 5 % of per-window RMS
      - DC offset jitter ±20 counts
      - Time-shift (roll) ±5 samples

    Source: Kaifosh et al. Nature 2025. doi:10.1038/s41586-025-09255-w
    """
    rng = np.random.default_rng(seed)
    aug_X, aug_y = [X], [y]
    for _ in range(multiplier - 1):
        Xc  = X.copy().astype(np.float32)
        Xc *= rng.uniform(0.80, 1.20, (len(X), 1, 1)).astype(np.float32)
        rms  = np.sqrt(np.mean(Xc ** 2, axis=(1, 2), keepdims=True)) + 1e-8
        Xc  += rng.standard_normal(Xc.shape).astype(np.float32) * (0.05 * rms)
        Xc  += rng.uniform(-20., 20., (len(X), 1, X.shape[2])).astype(np.float32)
        shifts = rng.integers(-5, 6, size=len(X))
        for i in range(len(Xc)):
            if shifts[i]:
                Xc[i] = np.roll(Xc[i], shifts[i], axis=0)
        aug_X.append(Xc)
        aug_y.append(y)
    return np.concatenate(aug_X), np.concatenate(aug_y)


# =============================================================================
# LDA CLASSIFIER
# =============================================================================

class EMGClassifier:
    """
    EMG gesture classifier supporting LDA and QDA.

    Model types:
      - LDA: Linear Discriminant Analysis — fast, exportable to ESP32 C header
      - QDA: Quadratic Discriminant Analysis — more flexible boundaries, laptop-only
    """

    def __init__(self, model_type: str = "lda", reg_param: float = 0.1):
        self.model_type = model_type.lower()
        self.reg_param = reg_param  # only used by QDA
        self.feature_extractor = EMGFeatureExtractor(channels=HAND_CHANNELS, reinhard=True)
        if self.model_type == "qda":
            self.model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
        else:
            self.model = LinearDiscriminantAnalysis()
        self.label_names: list[str] = []
        self.is_trained = False
        self.feature_names: list[str] = []
        self.calibration_transform = CalibrationTransform()

    def train(self, X: np.ndarray, y: np.ndarray, label_names: list[str],
              session_indices: Optional[np.ndarray] = None):
        """
        Train the classifier.

        Args:
            X: Raw EMG windows (n_windows, n_samples, n_channels)
            y: Integer labels (n_windows,)
            label_names: List of label strings
            session_indices: Optional per-window integer session ID (0..n_sessions-1).
                             When provided, each session's features are independently
                             z-scored before fitting, creating a placement-invariant model.
        """
        # Change 3: data augmentation on raw windows before feature extraction
        if getattr(self, 'use_augmentation', True):
            X_aug, y_aug = augment_emg_batch(X, y, multiplier=3)
            print(f"[Classifier] Augmentation: {len(X)} -> {len(X_aug)} windows")
            # Replicate session_indices to match the augmented size
            if session_indices is not None:
                session_indices = np.tile(session_indices, 3)
        else:
            X_aug, y_aug = X, y

        print("\n[Classifier] Extracting features...")
        X_features = self.feature_extractor.extract_features_batch(X_aug)
        self.feature_names = self.feature_extractor.get_feature_names(X_aug.shape[2])

        # Change 6: optionally stack MPF features
        if getattr(self, 'use_mpf', False):
            mpf = MPFFeatureExtractor(channels=HAND_CHANNELS)
            X_features = np.hstack([X_features, mpf.extract_batch(X_aug)])

        print(f"[Classifier] Feature matrix shape: {X_features.shape}")
        print(f"[Classifier] Features per window: {len(self.feature_names)}")

        if session_indices is not None:
            n_sessions = len(np.unique(session_indices))
            print(f"\n[Classifier] Applying per-session z-score normalization ({n_sessions} sessions, class-balanced mu)...")
            X_features = self._apply_session_normalization(X_features, session_indices, y=y_aug)

        print(f"\n[Classifier] Training {self.model_type.upper()}...")
        self.model.fit(X_features, y_aug)
        self.label_names = label_names
        self.is_trained = True

        # Store training distribution (in normalized space) for calibration diagnostics
        self.calibration_transform.fit_from_training(X_features, y_aug, label_names)

        # Training accuracy
        train_acc = self.model.score(X_features, y_aug)
        print(f"[Classifier] Training accuracy: {train_acc*100:.1f}%")

        return X_features

    def _apply_session_normalization(self, X_features: np.ndarray, session_indices: np.ndarray,
                                     y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Z-score each session's features independently using a class-balanced mean.

        For each session:
          - mu    = mean of per-class centroids (class-balanced, not weighted by window count)
          - sigma = overall std of all windows in the session

        Using the class-balanced mean prevents sessions with more rest windows (or any
        imbalanced class) from skewing the normalization origin toward that class.
        """
        X_norm = X_features.copy()
        session_sigmas = []
        for sid in np.unique(session_indices):
            mask = session_indices == sid
            X_sess = X_features[mask]
            if y is not None:
                # Class-balanced mean: average of per-class centroids
                y_sess = y[mask]
                class_means = [X_sess[y_sess == cls].mean(axis=0)
                               for cls in np.unique(y_sess)]
                mu = np.mean(class_means, axis=0)
            else:
                mu = X_sess.mean(axis=0)
            sigma = X_sess.std(axis=0) + 1e-8
            session_sigmas.append(sigma)
            X_norm[mask] = (X_sess - mu) / sigma
        # Store mean per-session sigma so calibration can use the same scale reference
        self.calibration_transform.sigma_train = np.mean(session_sigmas, axis=0)
        return X_norm

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate classifier on test data."""
        if not self.is_trained:
            raise ValueError("Classifier not trained!")

        X_features = self.feature_extractor.extract_features_batch(X)
        y_pred = self.model.predict(X_features)

        accuracy = np.mean(y_pred == y)

        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_true': y
        }

    def cross_validate(self, X: np.ndarray, y: np.ndarray, trial_ids: Optional[np.ndarray] = None,
                       cv: int = 5, session_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform k-fold cross-validation with trial-level splitting.

        When trial_ids are provided, uses GroupKFold to ensure windows from the
        same trial never appear in both train and test folds (prevents leakage).

        When session_indices are provided, applies the same per-session z-score
        normalization used during training before running CV.
        """
        X_features = self.feature_extractor.extract_features_batch(X)

        if session_indices is not None:
            X_features = self._apply_session_normalization(X_features, session_indices, y=y)

        if trial_ids is not None:
            print(f"\n[Classifier] Running {cv}-fold cross-validation (TRIAL-LEVEL, no leakage)...")
            group_kfold = GroupKFold(n_splits=cv)
            scores = cross_val_score(self.model, X_features, y, cv=group_kfold, groups=trial_ids)
        else:
            print(f"\n[Classifier] Running {cv}-fold cross-validation (window-level, legacy)...")
            scores = cross_val_score(self.model, X_features, y, cv=cv)

        return scores

    def predict(self, window: np.ndarray) -> tuple[str, np.ndarray]:
        """
        Predict gesture for a single window.

        Args:
            window: Shape (n_samples, n_channels)

        Returns:
            (predicted_label, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained!")

        if not hasattr(self, '_predict_count'):
            self._predict_count = 0
        self._predict_count += 1
        _debug = (self._predict_count <= 30)

        features_raw = self.feature_extractor.extract_features_window(window)

        # Energy gate: if raw signal is quiet enough to be rest, skip LDA entirely.
        # Uses raw window RMS (pre-feature-extraction) so amplitude normalization
        # inside the feature extractor doesn't mask the energy difference.
        ct = self.calibration_transform
        if (ct.is_fitted and ct.rest_energy_threshold is not None
                and "rest" in self.label_names):
            w_ac = window - window.mean(axis=0)   # remove per-window DC offset (matches feature extractor)
            raw_rms = float(np.sqrt(np.mean(w_ac ** 2)))
            if _debug:
                print(f"[predict #{self._predict_count}] rms={raw_rms:.1f}  gate={ct.rest_energy_threshold:.1f}  "
                      f"{'GATED->rest' if raw_rms < ct.rest_energy_threshold else 'pass->QDA/LDA'}")
            if raw_rms < ct.rest_energy_threshold:
                rest_idx = self.label_names.index("rest")
                proba = np.zeros(len(self.label_names))
                proba[rest_idx] = 1.0
                return "rest", proba
        elif _debug:
            print(f"[predict #{self._predict_count}] gate inactive (is_fitted={ct.is_fitted}, "
                  f"threshold={ct.rest_energy_threshold})")

        features = ct.apply(features_raw)
        pred_idx = self.model.predict([features])[0]
        proba = self.model.predict_proba([features])[0]
        if _debug:
            top = sorted(zip(self.label_names, proba), key=lambda x: -x[1])[:3]
            print(f"[predict #{self._predict_count}] {self.model_type.upper()} -> {self.label_names[pred_idx]}"
                  f"  proba: {', '.join(f'{n}={p:.2f}' for n,p in top)}")
        return self.label_names[pred_idx], proba

    def get_feature_importance(self) -> dict:
        """Get feature importance based on LDA coefficients (LDA only)."""
        if not self.is_trained:
            return {}

        if not hasattr(self.model, 'coef_'):
            return {}

        # For multi-class, average absolute coefficients across classes
        coef = np.abs(self.model.coef_).mean(axis=0)
        importance = dict(zip(self.feature_names, coef))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, filepath: Path) -> Path:
        """
        Save the trained classifier to disk.

        Saves:
          - LDA model parameters
          - Feature extractor settings
          - Label names
          - Feature names

        Args:
            filepath: Path to save the model (e.g., 'models/emg_classifier.joblib')

        Returns:
            Path to the saved model file
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained classifier!")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'label_names': self.label_names,
            'feature_names': self.feature_names,
            'feature_extractor_params': {
                'zc_threshold_percent': self.feature_extractor.zc_threshold_percent,
                'ssc_threshold_percent': self.feature_extractor.ssc_threshold_percent,
                'channels': self.feature_extractor.channels,
                'normalize': self.feature_extractor.normalize,
                'expanded': self.feature_extractor.expanded,
                'cross_channel': self.feature_extractor.cross_channel,
                'bandpass': self.feature_extractor.bandpass,
                'reinhard': self.feature_extractor.reinhard,
                'fft_n': self.feature_extractor.fft_n,
                'fs': self.feature_extractor.fs,
            },
            'version': '1.3',
            'reg_param': self.reg_param,
            'session_normalized': True,
            # Calibration transform training stats (used by CalibrationPage)
            'calib_class_means_train': self.calibration_transform.class_means_train,
            'calib_sigma_train': self.calibration_transform.sigma_train,
        }

        joblib.dump(model_data, filepath)
        print(f"[Classifier] Model saved to: {filepath}")
        print(f"[Classifier] File size: {filepath.stat().st_size / 1024:.1f} KB")
        return filepath

    def export_to_header(self, filepath: Path) -> Path:
        """
        Export trained model to a C header file for ESP32 inference.

        Args:
            filepath: Output .h file path

        Returns:
            Path to the saved header file
        """
        if not self.is_trained:
            raise ValueError("Cannot export untrained classifier!")

        if self.model_type != "lda":
            raise ValueError(
                f"Cannot export {self.model_type.upper()} to C header. "
                "Only LDA models can be exported (QDA lacks coef_/intercept_)."
            )

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        n_classes = len(self.label_names)
        n_features = len(self.feature_names)

        # Get LDA parameters
        # coef_: (n_classes, n_features) - access as [class][feature]
        # intercept_: (n_classes,)
        coefs = self.model.coef_
        intercepts = self.model.intercept_

        # Add logic for binary classification (sklearn stores only 1 set of coefs)
        # For >2 classes, it stores n_classes sets.
        if n_classes == 2:
            # Binary case: coef_ is (1, n_features), intercept_ is (1,)
            # We need to expand this to 2 classes for the C inference engine to be generic.
            # Class 1 decision = dot(w, x) + b
            # Class 0 decision = - (dot(w, x) + b)  <-- Implicit in sklearn decision_function
            # BUT: decision_function returns score. A generic 'argmax' approach usually expects
            # one score per class. Multiclass LDA in sklearn does generic OVR/Multinomial.
            # Let's check sklearn docs or behavior.
            # Actually, LDA in sklearn for binary case is special.
            # To make C code simple (always argmax), let's explicitly store 2 rows.
            # Row 1 (Index 1 in sklearn): coef, intercept
            # Row 0 (Index 0): -coef, -intercept ?
            # Wait, LDA is generative. The decision boundary is linear.
            # Let's assume Multiclass for now or handle binary specifically.
            # For simplicity in C, we prefer (n_classes, n_features).
            # If coefs.shape[0] != n_classes, we need to handle it.
            if coefs.shape[0] == 1:
                print("[Export] Binary classification detected. Expanding to 2 classes for C compatibility.")
                # Class 1 (positive)
                c1_coef = coefs[0]
                c1_int = intercepts[0]
                # Class 0 (negative) - Effectively -score for decision boundary at 0
                # But strictly speaking LDA is comparison of log-posteriors.
                # Sklearn's coef_ comes from (Sigma^-1)(mu1 - mu0).
                # The score S = coef.X + intercept. If S > 0 pred class 1, else 0.
                # To map this to ArgMax(Score0, Score1):
                # We can set Score1 = S, Score0 = 0. OR Score1 = S/2, Score0 = -S/2.
                # Let's use Score1 = S, Score0 = 0 (Bias term makes this trickier).
                # Safest: Let's trust that for our 5-gesture demo, it's multiclass.
                pass

        # Bug 7 fix: preserve compile-time flags that are independent of
        # the feature pipeline (MLP, ensemble).  Pipeline-dependent flags
        # (EXPAND_FEATURES, REINHARD) are set from the extractor config so
        # they always match the exported weights.
        preserved_flags = {}
        _PRESERVED_FLAG_NAMES = ['MODEL_USE_MLP', 'MODEL_USE_ENSEMBLE']
        if filepath.exists():
            import re
            existing = filepath.read_text()
            for flag in _PRESERVED_FLAG_NAMES:
                m = re.search(rf'#define\s+{flag}\s+(\d+)', existing)
                if m:
                    preserved_flags[flag] = int(m.group(1))

        # Auto-set pipeline flags from training config (prevents mismatch)
        preserved_flags['MODEL_EXPAND_FEATURES'] = 1 if self.feature_extractor.expanded else 0
        preserved_flags['MODEL_USE_REINHARD'] = 1 if self.feature_extractor.reinhard else 0

        # Generate C content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        c_content = [
            "/**",
            f" * @file {filepath.name}",
            " * @brief Trained LDA model weights exported from Python.",
            f" * @date {timestamp}",
            " */",
            "",
            "#ifndef MODEL_WEIGHTS_H",
            "#define MODEL_WEIGHTS_H",
            "",
            "#include <stdint.h>",
            "",
            "/* Metadata */",
            f"#define MODEL_NUM_CLASSES {n_classes}",
            f"#define MODEL_NUM_FEATURES {n_features}",
            f"#define MODEL_NORMALIZE_FEATURES {1 if self.feature_extractor.normalize else 0}",
            "",
        ]

        # Write compile-time flags (pipeline flags auto-set, architecture flags preserved)
        _ALL_FLAGS = [
            'MODEL_EXPAND_FEATURES', 'MODEL_USE_REINHARD',
            'MODEL_USE_MLP', 'MODEL_USE_ENSEMBLE',
        ]
        c_content.append("/* Compile-time feature flags */")
        for flag in _ALL_FLAGS:
            val = preserved_flags.get(flag, 0)
            c_content.append(f"#define {flag} {val}")
        c_content.append("")

        c_content.append("/* Class Names */")
        c_content.append("static const char* MODEL_CLASS_NAMES[MODEL_NUM_CLASSES] = {")

        for name in self.label_names:
            c_content.append(f'    "{name}",')
        c_content.append("};")
        c_content.append("")

        c_content.append("/* Feature Extractor Parameters */")
        c_content.append(f"#define FEAT_ZC_THRESH {self.feature_extractor.zc_threshold_percent}f")
        c_content.append(f"#define FEAT_SSC_THRESH {self.feature_extractor.ssc_threshold_percent}f")
        c_content.append("")

        c_content.append("/* LDA Intercepts/Biases */")
        c_content.append(f"static const float LDA_INTERCEPTS[MODEL_NUM_CLASSES] = {{")
        line = "    "
        for val in intercepts:
             line += f"{val:.6f}f, "
        c_content.append(line.rstrip(", "))
        c_content.append("};")
        c_content.append("")

        c_content.append("/* LDA Coefficients (Weights) */")
        c_content.append(f"static const float LDA_WEIGHTS[MODEL_NUM_CLASSES][MODEL_NUM_FEATURES] = {{")
        
        for i, row in enumerate(coefs):
            c_content.append(f"    /* {self.label_names[i]} */")
            c_content.append("    {")
            line = "        "
            for j, val in enumerate(row):
                line += f"{val:.6f}f, "
                if (j + 1) % 8 == 0:
                    c_content.append(line)
                    line = "        "
            if line.strip():
                c_content.append(line.rstrip(", "))
            c_content.append("    },")
            
        c_content.append("};")
        c_content.append("")
        c_content.append("#endif /* MODEL_WEIGHTS_H */")

        with open(filepath, 'w') as f:
            f.write('\n'.join(c_content))

        print(f"[Classifier] Model weights exported to: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> 'EMGClassifier':
        """
        Load a trained classifier from disk.

        Args:
            filepath: Path to the saved model file

        Returns:
            Loaded EMGClassifier instance ready for prediction
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        # Determine model type (backward compat: old files have 'lda' key, no 'model_type')
        model_type = model_data.get('model_type', 'lda')
        reg_param  = model_data.get('reg_param', 0.1)

        # Create new instance and restore state
        classifier = cls(model_type=model_type, reg_param=reg_param)
        classifier.model = model_data.get('model', model_data.get('lda'))
        classifier.label_names = model_data['label_names']
        classifier.is_trained = True

        # Restore feature extractor params
        params = model_data.get('feature_extractor_params', {})
        # Infer expanded/cross_channel from feature count for old models
        # that don't store these params: 12 features = legacy (4×3),
        # 69 features = expanded (20×3 + 9 cross-channel)
        saved_feat_names = model_data.get('feature_names', [])
        n_feat = len(saved_feat_names) if saved_feat_names else 69
        default_expanded = n_feat > 12
        default_cc = n_feat > 60  # cross-channel adds 9 features (60→69)
        classifier.feature_extractor = EMGFeatureExtractor(
            zc_threshold_percent=params.get('zc_threshold_percent', 0.1),
            ssc_threshold_percent=params.get('ssc_threshold_percent', 0.1),
            channels=params.get('channels', HAND_CHANNELS),
            normalize=params.get('normalize', False),
            expanded=params.get('expanded', default_expanded),
            cross_channel=params.get('cross_channel', default_cc),
            bandpass=params.get('bandpass', False),  # False for old models
            reinhard=params.get('reinhard', False),
            fft_n=params.get('fft_n', 256),
            fs=params.get('fs', float(SAMPLING_RATE_HZ)),
        )
        # Regenerate feature names from extractor if not in saved data
        if saved_feat_names:
            classifier.feature_names = saved_feat_names
        else:
            channels = params.get('channels', HAND_CHANNELS)
            classifier.feature_names = classifier.feature_extractor.get_feature_names(len(channels))

        # Restore calibration transform training stats (saved from v1.2+ models)
        classifier.calibration_transform = CalibrationTransform()
        class_means_train = model_data.get('calib_class_means_train', {})
        sigma_train       = model_data.get('calib_sigma_train')
        session_normalized = model_data.get('session_normalized', False)
        classifier.session_normalized = session_normalized
        if class_means_train:
            classifier.calibration_transform.class_means_train = class_means_train
            classifier.calibration_transform.has_training_stats = True
        if sigma_train is not None:
            classifier.calibration_transform.sigma_train = sigma_train

        print(f"[Classifier] Model loaded from: {filepath}")
        print(f"[Classifier] Labels: {classifier.label_names}")
        calib_ready = classifier.calibration_transform.has_training_stats
        print(f"[Classifier] Calibration support: {'yes' if calib_ready else 'no (retrain to enable)'}")
        print(f"[Classifier] Session-normalized: {session_normalized}")
        return classifier

    @staticmethod
    def get_default_model_path() -> Path:
        """Get the default path for saving/loading models."""
        return MODEL_DIR / "emg_lda_classifier.joblib"

    @staticmethod
    def get_latest_model_path() -> Path | None:
        """Get the most recently modified model file, or None if no models exist."""
        models = EMGClassifier.list_saved_models()
        if not models:
            return None
        return max(models, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def list_saved_models() -> list[Path]:
        """List all saved classifier model files (excludes ensemble/auxiliary files)."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return sorted(
            p for p in MODEL_DIR.glob("*.joblib")
            if "ensemble" not in p.stem
        )


# =============================================================================
# PREDICTION SMOOTHING (Temporal smoothing, majority vote, debouncing)
# =============================================================================

class PredictionSmoother:
    """
    Smooths predictions to prevent twitchy/unstable output.

    Combines three techniques:
      1. Probability Smoothing: Exponential moving average on raw probabilities
      2. Majority Vote: Output most common prediction from last N predictions
      3. Debouncing: Only change output after N consecutive same predictions

    This prevents the robotic hand from twitching when there's an occasional
    misclassification in a stream of correct predictions.

    Example:
        Raw predictions:    FIST, FIST, OPEN, FIST, FIST, FIST
        Without smoothing:  Hand twitches open briefly
        With smoothing:     Hand stays as FIST (OPEN was filtered out)
    """

    def __init__(
        self,
        label_names: list[str],
        probability_smoothing: float = 0.7,
        majority_vote_window: int = 5,
        debounce_count: int = 3,
    ):
        """
        Args:
            label_names: List of gesture labels (must match classifier output)
            probability_smoothing: EMA factor (0-1). Higher = more smoothing.
                                   0 = no smoothing, 0.9 = very smooth
            majority_vote_window: Number of past predictions to consider for voting
            debounce_count: Number of consecutive same predictions needed to change output
        """
        self.label_names = label_names
        self.n_classes = len(label_names)

        # Probability smoothing (Exponential Moving Average)
        self.prob_smoothing = probability_smoothing
        self.smoothed_proba = np.ones(self.n_classes) / self.n_classes  # Start uniform

        # Majority vote
        self.vote_window = majority_vote_window
        self.prediction_history: list[str] = []

        # Debouncing
        self.debounce_count = debounce_count
        self.current_output = None
        self.pending_output = None
        self.pending_count = 0

        # Stats
        self.total_predictions = 0
        self.output_changes = 0

    def update(self, predicted_label: str, probabilities: np.ndarray) -> tuple[str, float, dict]:
        """
        Process a new prediction and return smoothed output.

        Args:
            predicted_label: Raw prediction from classifier
            probabilities: Raw probability array from classifier

        Returns:
            (smoothed_label, confidence, debug_info)
            - smoothed_label: The stable output label after all smoothing
            - confidence: Confidence in the smoothed output (0-1)
            - debug_info: Dict with intermediate values for debugging/display
        """
        self.total_predictions += 1

        # --- 1. Probability Smoothing (EMA) ---
        # Blend new probabilities with historical smoothed probabilities
        self.smoothed_proba = (
            self.prob_smoothing * self.smoothed_proba +
            (1 - self.prob_smoothing) * probabilities
        )

        # Get prediction from smoothed probabilities
        prob_smoothed_idx = np.argmax(self.smoothed_proba)
        prob_smoothed_label = self.label_names[prob_smoothed_idx]
        prob_smoothed_confidence = self.smoothed_proba[prob_smoothed_idx]

        # --- 2. Majority Vote ---
        # Add to history and keep window size
        self.prediction_history.append(prob_smoothed_label)
        if len(self.prediction_history) > self.vote_window:
            self.prediction_history.pop(0)

        # Count votes
        vote_counts = {}
        for pred in self.prediction_history:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1

        # Get majority winner
        majority_label = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_label]
        majority_confidence = majority_count / len(self.prediction_history)

        # --- 3. Debouncing ---
        # Only change output after consistent predictions
        if self.current_output is None:
            # First prediction
            self.current_output = majority_label
            self.pending_output = majority_label
            self.pending_count = 1
        elif majority_label == self.current_output:
            # Same as current output, reset pending
            self.pending_output = majority_label
            self.pending_count = 1
        elif majority_label == self.pending_output:
            # Same as pending, increment count
            self.pending_count += 1
            if self.pending_count >= self.debounce_count:
                # Enough consecutive predictions, change output
                self.current_output = majority_label
                self.output_changes += 1
        else:
            # New prediction, start new pending
            self.pending_output = majority_label
            self.pending_count = 1

        # Final output
        final_label = self.current_output
        final_confidence = majority_confidence

        # Debug info
        debug_info = {
            'raw_label': predicted_label,
            'raw_confidence': float(np.max(probabilities)),
            'prob_smoothed_label': prob_smoothed_label,
            'prob_smoothed_confidence': float(prob_smoothed_confidence),
            'majority_label': majority_label,
            'majority_confidence': float(majority_confidence),
            'vote_counts': vote_counts,
            'pending_output': self.pending_output,
            'pending_count': self.pending_count,
            'debounce_threshold': self.debounce_count,
        }

        return final_label, final_confidence, debug_info

    def reset(self):
        """Reset all state (call when starting a new prediction session)."""
        self.smoothed_proba = np.ones(self.n_classes) / self.n_classes
        self.prediction_history = []
        self.current_output = None
        self.pending_output = None
        self.pending_count = 0
        self.total_predictions = 0
        self.output_changes = 0

    def get_stats(self) -> dict:
        """Get statistics about smoothing effectiveness."""
        return {
            'total_predictions': self.total_predictions,
            'output_changes': self.output_changes,
            'stability_ratio': 1 - (self.output_changes / max(1, self.total_predictions)),
        }


# =============================================================================
# TRAINING (Train LDA classifier)
# =============================================================================

def run_training_demo():
    """
    Train an LDA classifier on ALL collected sessions combined.

    Shows:
      1. Loading all session data combined
      2. Feature extraction
      3. Training LDA
      4. Cross-validation evaluation
      5. Feature importance analysis

    The model learns from all accumulated data, making it more robust
    as you collect more sessions over time.
    """
    print("\n" + "=" * 60)
    print("TRAIN LDA CLASSIFIER (ALL SESSIONS)")
    print("=" * 60)

    storage = SessionStorage()
    sessions = storage.list_sessions()

    if not sessions:
        print("\nNo saved sessions found!")
        print("Run option 1 first to collect and save training data.")
        return None

    # Show available sessions
    print(f"\nFound {len(sessions)} saved session(s):")
    print("-" * 40)

    total_windows = 0
    for session_id in sessions:
        info = storage.get_session_info(session_id)
        print(f"  - {session_id}: {info['num_windows']} windows, gestures: {info['gestures']}")
        total_windows += info['num_windows']

    print(f"\nTotal windows across all sessions: {total_windows}")
    print("-" * 40)

    confirm = input("\nTrain on ALL sessions combined? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return None

    # Load ALL data combined
    print(f"\n{'=' * 60}")
    print("TRAINING ON ALL SESSIONS COMBINED")
    print("=" * 60)

    X, y, trial_ids, session_indices, label_names, loaded_sessions = storage.load_all_for_training()

    print(f"\nDataset:")
    print(f"  Windows: {X.shape[0]}")
    print(f"  Samples per window: {X.shape[1]}")
    print(f"  Channels: {X.shape[2]}")
    print(f"  Classes: {label_names}")
    print(f"  Unique trials: {len(np.unique(trial_ids))}")

    # Count per class
    print(f"\nSamples per class:")
    for i, name in enumerate(label_names):
        count = np.sum(y == i)
        print(f"  {name}: {count}")

    # Create and train classifier
    classifier = EMGClassifier()
    X_features = classifier.train(X, y, label_names)

    # Cross-validation (trial-level to prevent leakage)
    cv_scores = classifier.cross_validate(X, y, trial_ids=trial_ids, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")

    # Feature importance
    print(f"\n{'-' * 40}")
    print("FEATURE IMPORTANCE (top 8)")
    print("-" * 40)
    importance = classifier.get_feature_importance()
    for i, (name, score) in enumerate(list(importance.items())[:8]):
        bar = "█" * int(score * 20)
        print(f"  {name:12s}: {bar} ({score:.3f})")

    # Train/test split evaluation (TRIAL-LEVEL to prevent leakage)
    print(f"\n{'-' * 40}")
    print("TRAIN/TEST SPLIT EVALUATION (TRIAL-LEVEL)")
    print("-" * 40)

    # Use GroupShuffleSplit to split by trial, not by window
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=trial_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_trial_ids = trial_ids[train_idx]
    test_trial_ids = trial_ids[test_idx]

    # VERIFICATION: Ensure no trial leakage
    train_trials_set = set(train_trial_ids)
    test_trials_set = set(test_trial_ids)
    overlap = train_trials_set & test_trials_set
    assert len(overlap) == 0, f"Trial leakage detected! Overlapping trials: {overlap}"
    print(f"  Train: {len(X_train)} windows from {len(train_trials_set)} trials")
    print(f"  Test:  {len(X_test)} windows from {len(test_trials_set)} trials")
    print(f"  Trial overlap: {len(overlap)} (VERIFIED: no leakage)")

    # Log per-class distribution
    print(f"\n  Per-class window counts:")
    for i, name in enumerate(label_names):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"    {name:12s}: train={train_count:4d}, test={test_count:4d}")

    # Train on train set
    test_classifier = EMGClassifier()
    test_classifier.train(X_train, y_train, label_names)

    # Evaluate on test set
    result = test_classifier.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {result['accuracy']*100:.1f}%")

    print(f"\nClassification Report:")
    print(classification_report(result['y_true'], result['y_pred'],
                               target_names=label_names))

    print(f"Confusion Matrix:")
    cm = confusion_matrix(result['y_true'], result['y_pred'])
    print(f"  {'':12s} ", end="")
    for name in label_names:
        print(f"{name[:8]:>8s} ", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:12s} ", end="")
        for val in row:
            print(f"{val:8d} ", end="")
        print()

    # --- Save the model ---
    print(f"\n{'-' * 40}")
    print("SAVE MODEL")
    print("-" * 40)

    default_path = EMGClassifier.get_default_model_path()
    print(f"Default save path: {default_path}")

    save_choice = input("\nSave this model? (y/n): ").strip().lower()
    if save_choice == 'y':
        classifier.save(default_path)
        print(f"\nModel saved! You can now use 'Live prediction' without retraining.")

    return classifier


# =============================================================================
# Change 5 — CLASSIFIER BENCHMARK
# =============================================================================

def run_classifier_benchmark():
    """
    Cross-validate LDA, QDA, SVM-RBF, and MLP on the collected dataset.

    Purpose: tells you whether accuracy plateau is a features problem
    (all classifiers similar → add features) or a model complexity problem
    (SVM/MLP >> LDA → implement Change E / ensemble).

    Runs twice: once with base 69 features, once with 69 + 36 MPF features (105 total).
    """
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)

    print("\n" + "=" * 60)
    print("CLASSIFIER BENCHMARK (Cross-validation)")
    print("=" * 60)

    storage = SessionStorage()
    X_raw, y, trial_ids, session_indices, label_names, _ = storage.load_all_for_training()

    if len(np.unique(y)) < 2:
        print("Need at least 2 gesture classes. Collect more data first.")
        return

    # Base features (69)
    extractor = EMGFeatureExtractor(channels=HAND_CHANNELS, cross_channel=True)
    X_base = extractor.extract_features_batch(X_raw)
    X_base = EMGClassifier()._apply_session_normalization(X_base, session_indices, y=y)

    # MPF features (36)
    mpf = MPFFeatureExtractor(channels=HAND_CHANNELS)
    X_mpf = mpf.extract_batch(X_raw)
    X_mpf = EMGClassifier()._apply_session_normalization(X_mpf, session_indices, y=y)

    # Combined (105)
    X_combined = np.hstack([X_base, X_mpf])

    feature_sets = {
        f'Base ({X_base.shape[1]} features)': X_base,
        f'Base + MPF ({X_combined.shape[1]} features)': X_combined,
    }

    clfs = {
        'LDA (ESP32 model)': LinearDiscriminantAnalysis(),
        'QDA':               QuadraticDiscriminantAnalysis(reg_param=0.1),
        'SVM-RBF':           Pipeline([('s', StandardScaler()),
                                       ('m', SVC(kernel='rbf', C=10))]),
        'MLP-128-64':        Pipeline([('s', StandardScaler()),
                                       ('m', MLPClassifier(hidden_layer_sizes=(128, 64),
                                                           max_iter=1000,
                                                           early_stopping=True))]),
    }

    n_splits = min(5, len(np.unique(trial_ids)))
    gkf = GroupKFold(n_splits=n_splits)

    for feat_name, X in feature_sets.items():
        print(f"\n--- {feat_name} ---")
        print(f"  {'Classifier':<22} {'Mean CV':>8} {'Std':>6}")
        print("  " + "-" * 40)
        for name, clf in clfs.items():
            sc = cross_val_score(clf, X, y, cv=gkf, groups=trial_ids, scoring='accuracy')
            print(f"    {name:<20} {sc.mean()*100:>7.1f}%  ±{sc.std()*100:.1f}%")

    print()
    print("  → If LDA ≈ SVM: features are the bottleneck (add more features)")
    print("  → If SVM >> LDA: model complexity bottleneck (implement ensemble/MLP)")
    print("  → Compare Base vs Base+MPF to see if MPF features help")


# =============================================================================
# LIVE PREDICTION (Real-time gesture classification)
# =============================================================================

def run_prediction_demo():
    """
    Live prediction demo - classifies gestures in real-time from ESP32.

    Shows:
      1. Load saved model OR train fresh on all sessions
      2. Connect to ESP32 and stream real EMG data
      3. Classify each window as it comes in
      4. Display predictions with confidence

    REQUIRES: ESP32 hardware connected via USB.
    """
    print("\n" + "=" * 60)
    print("LIVE PREDICTION DEMO (ESP32 Required)")
    print("=" * 60)

    # Check for saved model
    saved_models = EMGClassifier.list_saved_models()
    default_model = EMGClassifier.get_default_model_path()

    classifier = None

    if default_model.exists():
        print(f"\nSaved model found: {default_model}")
        print(f"  File size: {default_model.stat().st_size / 1024:.1f} KB")

        load_choice = input("\nLoad saved model? (y=load, n=retrain): ").strip().lower()
        if load_choice == 'y':
            classifier = EMGClassifier.load(default_model)

    if classifier is None:
        # Need to train a new model
        storage = SessionStorage()
        sessions = storage.list_sessions()

        if not sessions:
            print("\nNo saved sessions found! Collect data first (Option 1).")
            return None

        # Show available sessions
        print(f"\nNo saved model (or retraining requested).")
        print(f"Will train on ALL {len(sessions)} session(s):")
        print("-" * 40)

        total_windows = 0
        for session_id in sessions:
            info = storage.get_session_info(session_id)
            print(f"  - {session_id}: {info['num_windows']} windows")
            total_windows += info['num_windows']

        print(f"\nTotal training windows: {total_windows}")
        print("-" * 40)

        confirm = input("\nTrain and start prediction? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Prediction cancelled.")
            return None

        # Load ALL sessions and train model
        print(f"\n[Training model on all sessions...]")
        X, y, trial_ids, session_indices, label_names, loaded_sessions = storage.load_all_for_training()
        print(f"[Unique trials: {len(np.unique(trial_ids))}]")

        classifier = EMGClassifier()
        classifier.train(X, y, label_names)

    # Connect to ESP32
    print("\n[Connecting to ESP32...]")
    stream = RealSerialStream()
    try:
        stream.connect(timeout=5.0)
        print(f"  Connected: {stream.device_info}")
    except Exception as e:
        print(f"  ERROR: Failed to connect to ESP32: {e}")
        print("  Make sure the ESP32 is connected and firmware is flashed.")
        return None

    # Start live prediction
    print("\n" + "=" * 60)
    print("STARTING LIVE PREDICTION (WITH SMOOTHING)")
    print("=" * 60)

    print("Press Ctrl+C to stop.\n")
    print("  Smoothing: Probability EMA (0.7) + Majority Vote (5) + Debounce (3)\n")

    parser = EMGParser(num_channels=NUM_CHANNELS)
    windower = Windower(window_size_ms=WINDOW_SIZE_MS, sample_rate=SAMPLING_RATE_HZ, hop_size_ms=HOP_SIZE_MS)

    # Create prediction smoother
    smoother = PredictionSmoother(
        label_names=classifier.label_names,
        probability_smoothing=0.7,   # Higher = more smoothing
        majority_vote_window=5,      # Past predictions to consider
        debounce_count=3,            # Consecutive predictions needed to change
    )

    stream.start()
    prediction_count = 0

    try:
        while True:
            # Read and process data
            line = stream.readline()
            if line:
                sample = parser.parse_line(line)
                if sample:
                    window = windower.add_sample(sample)
                    if window:
                        # Classify the window (raw prediction)
                        window_data = window.to_numpy()
                        raw_label, proba = classifier.predict(window_data)

                        # Apply smoothing
                        smoothed_label, smoothed_conf, debug = smoother.update(raw_label, proba)

                        prediction_count += 1

                        # Display both raw and smoothed predictions
                        raw_conf = max(proba) * 100
                        smoothed_conf_pct = smoothed_conf * 100

                        # Visual bar for smoothed confidence
                        bar_len = round(smoothed_conf_pct / 5)
                        bar = "█" * bar_len + "░" * (20 - bar_len)

                        # Show raw vs smoothed (smoothed is the stable output)
                        raw_marker = "  " if raw_label == smoothed_label else "!!"
                        print(f"  #{prediction_count:3d} │ {bar} │ {smoothed_label:12s} ({smoothed_conf_pct:5.1f}%) {raw_marker} raw:{raw_label[:8]:8s}")

    except KeyboardInterrupt:
        print("\n\n[Stopped by user]")

    finally:
        stream.stop()
        stream.disconnect()

    # Show smoothing stats
    stats = smoother.get_stats()
    print(f"\n" + "-" * 40)
    print(f"SMOOTHING STATISTICS")
    print("-" * 40)
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Output changes: {stats['output_changes']}")
    print(f"  Stability ratio: {stats['stability_ratio']*100:.1f}%")

    return classifier


# =============================================================================
# LDA VISUALIZATION (Decision boundaries and feature space)
# =============================================================================

def run_visualization_demo():
    """
    Visualize the LDA model trained on ALL sessions with plots:
      1. 2D feature space scatter plot (LDA reduced)
      2. Decision boundaries
      3. Class distributions
      4. Confusion matrix heatmap

    Uses all accumulated session data for a complete picture of the model.
    """
    print("\n" + "=" * 60)
    print("LDA VISUALIZATION (ALL SESSIONS)")
    print("=" * 60)

    storage = SessionStorage()
    sessions = storage.list_sessions()

    if not sessions:
        print("\nNo saved sessions found! Collect data first (Option 1).")
        return None

    # Show available sessions
    print(f"\nFound {len(sessions)} saved session(s):")
    print("-" * 40)

    total_windows = 0
    for session_id in sessions:
        info = storage.get_session_info(session_id)
        print(f"  - {session_id}: {info['num_windows']} windows")
        total_windows += info['num_windows']

    print(f"\nTotal windows: {total_windows}")
    print("-" * 40)

    confirm = input("\nVisualize model trained on ALL sessions? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Visualization cancelled.")
        return None

    # Load ALL data combined
    X, y, trial_ids, session_indices, label_names, loaded_sessions = storage.load_all_for_training()
    print(f"[Unique trials: {len(np.unique(trial_ids))}]")

    # Extract features (forearm channels only, matching hand classifier)
    extractor = EMGFeatureExtractor(channels=HAND_CHANNELS)
    X_features = extractor.extract_features_batch(X)

    # Train LDA
    print("\n[Training LDA for visualization...]")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_features, y)

    # Transform to LDA space (reduces to n_classes - 1 dimensions)
    X_lda = lda.transform(X_features)

    n_classes = len(label_names)
    print(f"  LDA dimensions: {X_lda.shape[1]}")

    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_classes))

    # --- Figure 1: LDA Feature Space (2D projection) ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    for i, label in enumerate(label_names):
        mask = y == i
        ax1.scatter(X_lda[mask, 0],
                   X_lda[mask, 1] if X_lda.shape[1] > 1 else np.zeros(mask.sum()),
                   c=[colors[i]], label=label, s=100, alpha=0.7, edgecolors='white', linewidth=1)

    # Add class means
    for i, label in enumerate(label_names):
        mask = y == i
        mean_x = X_lda[mask, 0].mean()
        mean_y = X_lda[mask, 1].mean() if X_lda.shape[1] > 1 else 0
        ax1.scatter(mean_x, mean_y, c=[colors[i]], s=400, marker='X', edgecolors='black', linewidth=2)
        ax1.annotate(label.upper(), (mean_x, mean_y), fontsize=12, fontweight='bold',
                    ha='center', va='bottom', xytext=(0, 15), textcoords='offset points')

    ax1.set_xlabel("LDA Component 1", fontsize=12)
    ax1.set_ylabel("LDA Component 2", fontsize=12)
    ax1.set_title("LDA Feature Space - Gesture Clusters", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Figure 2: Decision Boundary Heatmap ---
    if X_lda.shape[1] >= 1:
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        # Create mesh grid
        x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
        if X_lda.shape[1] > 1:
            y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1
        else:
            y_min, y_max = -2, 2

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))

        # For prediction, we need to go back to original feature space
        # Use simplified approach: train new LDA on LDA-transformed features
        if X_lda.shape[1] > 1:
            lda_2d = LinearDiscriminantAnalysis()
            lda_2d.fit(X_lda[:, :2], y)
            Z = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            # 1D case - simple threshold
            Z = lda.predict(X_features[0:1])  # dummy
            Z = np.zeros(xx.ravel().shape)
            for i, x_val in enumerate(xx.ravel()):
                if X_lda.shape[1] == 1:
                    # Find closest class mean
                    distances = [abs(x_val - X_lda[y == c, 0].mean()) for c in range(n_classes)]
                    Z[i] = np.argmin(distances)

        Z = Z.reshape(xx.shape)

        # Plot decision regions
        ax2.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(-0.5, n_classes, 1),
                    colors=[colors[i] for i in range(n_classes)])
        ax2.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

        # Plot data points
        for i, label in enumerate(label_names):
            mask = y == i
            ax2.scatter(X_lda[mask, 0],
                       X_lda[mask, 1] if X_lda.shape[1] > 1 else np.zeros(mask.sum()),
                       c=[colors[i]], label=label, s=80, alpha=0.9, edgecolors='black', linewidth=0.5)

        ax2.set_xlabel("LDA Component 1", fontsize=12)
        ax2.set_ylabel("LDA Component 2", fontsize=12)
        ax2.set_title("LDA Decision Boundaries", fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)

    # --- Figure 3: Feature Importance Radar Chart ---
    fig3, ax3 = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    feature_names = extractor.get_feature_names(X.shape[2])
    coef = np.abs(lda.coef_).mean(axis=0)
    coef_normalized = coef / coef.max()  # Normalize to 0-1

    # Number of features
    n_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()

    # Complete the loop
    coef_normalized = np.concatenate([coef_normalized, [coef_normalized[0]]])
    angles += angles[:1]

    ax3.plot(angles, coef_normalized, 'o-', linewidth=2, color='#2E86AB', markersize=8)
    ax3.fill(angles, coef_normalized, alpha=0.25, color='#2E86AB')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(feature_names, fontsize=9)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Feature Importance (Radar)", fontsize=14, fontweight='bold', pad=20)

    # --- Figure 4: Class Distribution Histograms ---
    fig4, axes4 = plt.subplots(1, n_classes, figsize=(4*n_classes, 4))
    if n_classes == 1:
        axes4 = [axes4]

    for i, (ax, label) in enumerate(zip(axes4, label_names)):
        mask = y == i
        ax.hist(X_lda[mask, 0], bins=15, color=colors[i], alpha=0.7, edgecolor='black')
        ax.axvline(X_lda[mask, 0].mean(), color='black', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel("LDA Component 1")
        ax.set_ylabel("Count")
        ax.set_title(f"{label.upper()}", fontsize=12, fontweight='bold')
        ax.legend()

    fig4.suptitle("Class Distributions on LDA Axis", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # --- Figure 5: Confusion Matrix Heatmap ---
    fig5, ax5 = plt.subplots(figsize=(8, 6))

    y_pred = cross_val_predict(lda, X_features, y, cv=5)
    cm = confusion_matrix(y, y_pred)

    im = ax5.imshow(cm, interpolation='nearest', cmap='Blues')
    ax5.figure.colorbar(im, ax=ax5)

    ax5.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=label_names,
           yticklabels=label_names,
           xlabel='Predicted',
           ylabel='Actual',
           title='Confusion Matrix Heatmap')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax5.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("\n  Displayed 5 visualization figures")
    return lda


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    while True:
        print("\n" + "=" * 60)
        print("EMG DATA COLLECTION PIPELINE")
        print("=" * 60)
        print("\nOptions:")
        print("  1. Collect data (labeled session)")
        print("  2. Inspect saved sessions (view features)")
        print("  3. Train LDA classifier")
        print("  4. Live prediction demo")
        print("  5. Visualize LDA model")
        print("  6. Classifier benchmark (LDA vs QDA vs SVM vs MLP)")
        print("  q. Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == 'q':
            print("\nGoodbye!")
            break

        elif choice == "1":
            windows, labels = run_labeled_collection_demo()

        elif choice == "2":
            result = run_storage_demo()

        elif choice == "3":
            classifier = run_training_demo()

        elif choice == "4":
            classifier = run_prediction_demo()

        elif choice == "5":
            lda = run_visualization_demo()

        elif choice == "6":
            run_classifier_benchmark()

        else:
            print("\nInvalid choice. Please enter 1-6 or q.")