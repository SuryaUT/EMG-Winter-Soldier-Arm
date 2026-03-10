/**
 * @file bicep.h
 * @brief Bicep channel (ch3) subsystem — Phase 1: binary flex/unflex detector.
 *
 * Implements a simple RMS threshold detector with hysteresis for bicep activation.
 * ch3 data flows through the same IIR bandpass filter and circular buffer as the
 * hand gesture channels (via inference_get_bicep_rms()), so no separate ADC read
 * is required.
 *
 * Usage:
 *   1. On startup: bicep_load_threshold(&thresh) — restore persisted threshold
 *   2. After 3 s of relaxed rest:
 *        bicep_calibrate(raw_ch3_samples, n_samples)  — sets + saves threshold
 *   3. Every 25 ms hop:
 *        bicep_state_t state = bicep_detect();
 */

#ifndef BICEP_H
#define BICEP_H

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Bicep activation state.
 */
typedef enum {
    BICEP_STATE_REST = 0,
    BICEP_STATE_FLEX = 1,
} bicep_state_t;

/**
 * @brief Calibrate bicep threshold from REST data.
 *
 * Computes rest-RMS over the provided samples, then sets the internal
 * detection threshold to rest_rms × BICEP_FLEX_MULTIPLIER.
 *
 * @param ch3_samples  Raw ADC / mV values from the bicep channel.
 * @param n_samples    Number of samples provided.
 * @return Computed threshold in the same units as ch3_samples.
 */
float bicep_calibrate(const uint16_t *ch3_samples, int n_samples);

/**
 * @brief Detect current bicep state from the latest window.
 *
 * Uses inference_get_bicep_rms(BICEP_WINDOW_SAMPLES) internally, so
 * inference_add_sample() must have been called to fill the buffer first.
 *
 * @return BICEP_STATE_FLEX or BICEP_STATE_REST.
 */
bicep_state_t bicep_detect(void);

/**
 * @brief Persist the current threshold to NVS.
 *
 * @param threshold_mv  Threshold value to save (in mV / same units as bicep RMS).
 * @return true on success.
 */
bool bicep_save_threshold(float threshold_mv);

/**
 * @brief Load the persisted threshold from NVS.
 *
 * @param threshold_mv_out  Output pointer; untouched on failure.
 * @return true if a valid threshold was loaded.
 */
bool bicep_load_threshold(float *threshold_mv_out);

/**
 * @brief Set the detection threshold directly (without NVS save).
 */
void bicep_set_threshold(float threshold_mv);

/**
 * @brief Return the current threshold (0 if not calibrated).
 */
float bicep_get_threshold(void);

/**
 * @brief Calibrate bicep threshold from the filtered inference buffer.
 *
 * Uses inference_get_bicep_rms() to read from the bandpass-filtered
 * circular buffer — the same data source that bicep_detect() uses.
 * The old bicep_calibrate() accepts raw uint16_t ADC values, which are
 * in a different domain (includes DC offset) and produce unusable thresholds.
 *
 * Call this after the inference buffer has been filled with ≥ n_samples
 * of rest data via inference_add_sample().
 *
 * @param n_samples  Number of recent buffer samples to use for RMS.
 *                   Clamped to INFERENCE_WINDOW_SIZE internally.
 * @return Computed threshold (same units as bicep_detect sees).
 */
float bicep_calibrate_from_buffer(int n_samples);

#endif /* BICEP_H */
