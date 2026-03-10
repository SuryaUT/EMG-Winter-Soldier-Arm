/**
 * @file inference_mlp.h
 * @brief int8 MLP inference via TFLite Micro (Change E).
 *
 * Enable by:
 *   1. Setting MODEL_USE_MLP 1 in model_weights.h.
 *   2. Running train_mlp_tflite.py to generate emg_model_data.cc.
 *   3. Adding TFLite Micro to platformio.ini lib_deps.
 *
 * When MODEL_USE_MLP 0, both functions are empty stubs that compile without
 * TFLite Micro headers.
 */

#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialise TFLite Micro interpreter and allocate tensors.
 *        Must be called before inference_mlp_predict().
 *        No-op when MODEL_USE_MLP 0.
 */
void inference_mlp_init(void);

/**
 * @brief Run one forward pass through the int8 MLP.
 *
 * @param features   Float32 feature vector (same order as Python extractor).
 * @param n_feat     Number of features (must match model input size).
 * @param conf_out   Output: winning class softmax probability [0,1].
 * @return Winning class index.  Returns 0 when MODEL_USE_MLP 0.
 */
int inference_mlp_predict(const float *features, int n_feat, float *conf_out);

#ifdef __cplusplus
}
#endif
