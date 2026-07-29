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
 * @brief Drive the bicep servo to match the detected state (edge-triggered).
 *
 * Commands the bicep servo (JOINT_BICEP -> PCA channel 6) to BICEP_FLEX_ANGLE
 * on FLEX and BICEP_REST_ANGLE on REST. The servo is written only when the
 * state changes from the previously applied state, avoiding redundant I2C
 * writes every inference hop. Typical use: bicep_apply(bicep_detect()).
 *
 * @param state  Desired bicep state (from bicep_detect()).
 */
void bicep_apply(bicep_state_t state);

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

/*******************************************************************************
 * Proportional control (continuous flex)
 *
 * Instead of snapping between two angles, the servo tracks muscle effort:
 * activation level in [0,1] is derived from the live windowed RMS, normalized
 * against a two-point calibration (rest RMS floor, max-flex RMS ceiling). A
 * half-strength contraction holds the arm roughly halfway.
 ******************************************************************************/

/**
 * @brief Current windowed bicep RMS (same window bicep_detect() uses).
 *
 * Convenience wrapper over inference_get_bicep_rms() so callers don't need to
 * know the internal window size. Used during two-point calibration.
 */
float bicep_current_rms(void);

/**
 * @brief Set the proportional calibration reference points.
 *
 * @param rest_rms  Windowed RMS with the muscle relaxed (lower bound → 0.0).
 * @param max_rms   Windowed RMS at a hard, sustained flex (upper bound → 1.0).
 */
void bicep_set_proportional(float rest_rms, float max_rms);

/**
 * @brief Persist the proportional calibration (rest + max RMS) to NVS.
 * @return true on success.
 */
bool bicep_save_proportional(float rest_rms, float max_rms);

/**
 * @brief Load the persisted proportional calibration from NVS.
 *
 * @param rest_rms_out  Output; untouched on failure.
 * @param max_rms_out   Output; untouched on failure.
 * @return true if a valid (max > rest) calibration was loaded.
 */
bool bicep_load_proportional(float *rest_rms_out, float *max_rms_out);

/**
 * @brief Current bicep activation level, clamped to [0,1].
 *
 * 0.0 = at/below the rest floor, 1.0 = at/above the max ceiling. Reads the
 * latest windowed RMS and normalizes by (max - rest). A dead-zone near rest
 * forces a relaxed muscle to read exactly 0. Returns 0 if not calibrated.
 */
float bicep_get_level(void);

/**
 * @brief Drive the bicep servo proportionally to muscle effort (call every hop).
 *
 * Smooths bicep_get_level() with an EMA to suppress EMG jitter, rate-limits how
 * fast the target angle may slew, and writes the servo only when the angle has
 * moved meaningfully (keeps I2C traffic down while holding a pose).
 */
void bicep_apply_proportional(void);

#endif /* BICEP_H */
