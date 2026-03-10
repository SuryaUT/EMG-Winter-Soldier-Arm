/**
 * @file calibration.h
 * @brief NVS-backed z-score feature calibration for on-device EMG inference.
 *
 * Change D: Stores per-feature mean and std computed from a short REST session
 * in ESP32 non-volatile storage (NVS). At inference time, calibration_apply()
 * z-scores each feature vector before the LDA classifier sees it. This removes
 * day-to-day electrode placement drift without retraining the model.
 *
 * Typical workflow:
 *   1. calibration_init()        — called once at startup; loads from NVS
 *   2. calibration_update()      — called after collecting ~3s of REST windows
 *   3. calibration_apply(feat)   — called every inference hop in inference.c
 */

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <stdbool.h>
#include <stdint.h>

/* Maximum supported feature vector length.
 * 96 > 69 (expanded) > 12 (legacy) — gives headroom for future expansion. */
#define CALIB_MAX_FEATURES 96

/**
 * @brief Initialise NVS and load stored calibration statistics.
 *
 * Must be called once before any inference starts (e.g., in app_main).
 * @return true  Calibration data found and loaded.
 * @return false No stored data; calibration_apply() will be a no-op.
 */
bool calibration_init(void);

/**
 * @brief Apply stored z-score calibration to a feature vector in-place.
 *
 * x_i_out = (x_i - mean_i) / std_i
 *
 * No-op if calibration is not valid (calibration_is_valid() == false).
 * @param feat Feature vector of length n_feat (set during calibration_update).
 */
void calibration_apply(float *feat);

/**
 * @brief Compute and persist calibration statistics from REST EMG windows.
 *
 * Computes per-feature mean and std over n_windows windows, stores to NVS.
 * After this call, calibration_apply() uses the new statistics.
 *
 * @param X_flat  Flattened feature array [n_windows × n_feat], row-major.
 * @param n_windows  Number of windows (minimum 10).
 * @param n_feat     Feature vector length (≤ CALIB_MAX_FEATURES).
 * @return true on success, false if inputs are invalid or NVS write fails.
 */
bool calibration_update(const float *X_flat, int n_windows, int n_feat);

/**
 * @brief Clear calibration state (in-memory only; does not erase NVS).
 */
void calibration_reset(void);

/**
 * @brief Check whether valid calibration statistics are loaded.
 */
bool calibration_is_valid(void);

#endif /* CALIBRATION_H */
