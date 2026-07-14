/**
 * @file inference_ensemble.h
 * @brief 3-specialist-LDA + meta-LDA ensemble inference pipeline (Change F).
 *
 * Requires:
 *   - Change 1 expanded features (MODEL_EXPAND_FEATURES 1)
 *   - Change 7 training (train_ensemble.py) to generate model_weights_ensemble.h
 *
 * Enable by setting MODEL_USE_ENSEMBLE 1 in model_weights.h. This module is
 * STATELESS: it only computes probabilities. Smoothing/hysteresis and the
 * averaging across LDA/ensemble/MLP live in main.c (vote_postprocess).
 */

#pragma once
#include <stdbool.h>

/**
 * @brief Initialise the ensemble. Currently a no-op — the ensemble is stateless.
 *        Kept so callers have a stable init point.
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
