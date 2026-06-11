import h5py, json, numpy as np

HDF5_PATH  = "collected_data/updated010_20260214_204204.hdf5"
CALIB_SAMPLES = 2000
HOP    = 25   # firmware stride (samples)
WINDOW = 150  # firmware window size (samples)

with h5py.File(HDF5_PATH, "r") as f:
    gestures    = json.loads(f.attrs["gestures"])   # ["open","fist","hook_em","thumbs_up"]
    timestamps  = f["raw_samples/timestamps"][:]    # absolute ms, one per sample
    trial_ids   = f["windows/trial_ids"][:]         # gesture index per HDF5 window
    start_times = f["windows/start_times"][:]       # absolute ms per HDF5 window

t0 = timestamps[0]
n_samples = len(timestamps)

expected = []

window_sample_indices = start_times - t0

for fw_start in range(CALIB_SAMPLES, n_samples - WINDOW, HOP):
    nearest = np.argmin(np.abs(window_sample_indices - fw_start))
    if np.abs(window_sample_indices[nearest] - fw_start) <= HOP:
        expected.append(gestures[trial_ids[nearest]])
    else:
        expected.append("rest")