/**
 * @file inference.h
 * @brief On-device inference engine for EMG gesture recognition.
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdbool.h>
#include <stdint.h>

// --- Configuration (must match Python WINDOW_SIZE_MS / HOP_SIZE_MS) ---
#define INFERENCE_WINDOW_SIZE 150 // Window size in samples (150ms at 1kHz)
#define INFERENCE_HOP_SIZE     25 // Hop/stride in samples (25ms at 1kHz)
#define NUM_CHANNELS 4            // Total EMG channels (buffer stores all)
#define HAND_NUM_CHANNELS 3       // Forearm channels for hand classifier (ch0-ch2)

/**
 * @brief Initialize the inference engine.
 */
void inference_init(void);

/**
 * @brief Add a sample to the inference buffer.
 *
 * @param channels Array of 4 channel values (raw ADC)
 * @return true if a full window is ready for processing
 */
bool inference_add_sample(uint16_t *channels);

/**
 * @brief Run inference on the current window.
 *
 * @param confidence Output pointer for confidence score (0.0 - 1.0)
 * @return Detected class index (-1 if error)
 */
int inference_predict(float *confidence);

/**
 * @brief Get the name of a class index.
 */
const char *inference_get_class_name(int class_idx);

/**
 * @brief Map class index to gesture_t enum.
 */
int inference_get_gesture_enum(int class_idx);

/**
 * @brief Map a gesture name string directly to gesture_t enum value.
 *
 * Used by the laptop-predict path to convert the name sent by live_predict.py
 * into a gesture_t without needing a class index.
 *
 * @param name Lowercase gesture name, e.g. "fist", "rest", "open"
 * @return gesture_t value, or GESTURE_NONE if unrecognised
 */
int inference_get_gesture_by_name(const char *name);

/**
 * @brief Compute LDA softmax probabilities without smoothing/voting/debounce.
 *
 * Used by the multi-model voting path in main.c.  The caller is responsible
 * for post-processing (EMA, majority vote, debounce).
 *
 * @param features   Calibrated feature vector (MODEL_NUM_FEATURES floats).
 * @param proba_out  Output probability array (MODEL_NUM_CLASSES floats).
 */
void inference_predict_raw(const float *features, float *proba_out);

/**
 * @brief Extract and calibrate features from the current window.
 *
 * Dispatches to compute_features() or compute_features_expanded() depending
 * on MODEL_EXPAND_FEATURES, then applies calibration_apply().  The resulting
 * float array is identical to what inference_predict() uses internally.
 *
 * Called by inference_ensemble.c so that the ensemble path does not duplicate
 * the feature-extraction logic.
 *
 * @param features_out  Caller-allocated array of MODEL_NUM_FEATURES floats.
 */
void inference_extract_features(float *features_out);

/**
 * @brief Compute RMS of the last n_samples from channel 3 (bicep) in the
 *        inference circular buffer.
 *
 * Used by the bicep subsystem to obtain current activation level without
 * exposing the internal window_buffer.
 *
 * @param n_samples Number of samples to include (clamped to INFERENCE_WINDOW_SIZE).
 * @return RMS value in the same units as the filtered buffer (mV after Change B).
 *         Returns 0 if the buffer is not yet filled.
 */
float inference_get_bicep_rms(int n_samples);

#endif /* INFERENCE_H */
