/**
 * @file inference_ensemble.h
 * @brief 3-specialist-LDA + meta-LDA ensemble inference pipeline (Change F).
 *
 * Requires:
 *   - Change 1 expanded features (MODEL_EXPAND_FEATURES 1)
 *   - Change 7 training (train_ensemble.py) to generate model_weights_ensemble.h
 *   - Change E MLP (MODEL_USE_MLP 1) for confidence-cascade fallback
 *
 * Enable by setting MODEL_USE_ENSEMBLE 1 in model_weights.h and calling
 * inference_ensemble_init() from app_main() instead of (or alongside) inference_init().
 */

#pragma once
#include <stdbool.h>

/**
 * @brief Initialise ensemble state (EMA, vote history, debounce).
 *        Must be called before inference_ensemble_predict().
 */
void inference_ensemble_init(void);

/**
 * @brief Compute ensemble probabilities without smoothing/voting/debounce.
 *
 * Runs the 3 specialist LDAs + meta-LDA stacker and writes the raw meta-LDA
 * probabilities to proba_out.  Used by the multi-model voting path in main.c.
 *
 * @param features   Calibrated feature vector (MODEL_NUM_FEATURES floats).
 * @param proba_out  Output probability array (MODEL_NUM_CLASSES floats).
 */
void inference_ensemble_predict_raw(const float *features, float *proba_out);

/**
 * @brief Run one inference hop through the full ensemble pipeline.
 *
 * Internally calls inference_extract_features() to pull the latest window,
 * routes through the three specialist LDAs, the meta-LDA stacker, EMA
 * smoothing, majority vote, and debounce.
 *
 * @param confidence  Output: winning class smoothed probability [0,1].
 * @return Gesture enum value (same as inference_predict() return).
 */
int inference_ensemble_predict(float *confidence);
