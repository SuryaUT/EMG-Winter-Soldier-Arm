/**
 * @file calibration.c
 * @brief NVS-backed z-score feature calibration (Change D).
 */

#include "calibration.h"
#include "model_weights.h"   /* MODEL_FEAT_STD / MODEL_HAS_TRAIN_STD (sigma_train) */
#include "nvs_flash.h"
#include "nvs.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#define NVS_NAMESPACE "emg_calib"
#define NVS_KEY_MEAN  "feat_mean"
#define NVS_KEY_STD   "feat_std"
#define NVS_KEY_NFEAT "n_feat"
#define NVS_KEY_VALID "calib_ok"

static float s_mean[CALIB_MAX_FEATURES];
static float s_std[CALIB_MAX_FEATURES];
static int   s_n_feat = 0;
static bool  s_valid  = false;

bool calibration_init(void) {
    /* Standard NVS flash initialisation boilerplate */
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) {
        printf("[Calib] No NVS partition found — identity transform active\n");
        return false;
    }

    uint8_t  valid   = 0;
    int32_t  n_feat  = 0;
    size_t   mean_sz = sizeof(s_mean);
    size_t   std_sz  = sizeof(s_std);

    bool ok = (nvs_get_u8  (h, NVS_KEY_VALID, &valid)                    == ESP_OK) &&
              (valid == 1)                                                             &&
              (nvs_get_i32 (h, NVS_KEY_NFEAT, &n_feat)                   == ESP_OK) &&
              (n_feat > 0 && n_feat <= CALIB_MAX_FEATURES)                            &&
              (nvs_get_blob(h, NVS_KEY_MEAN, s_mean, &mean_sz)           == ESP_OK) &&
              (nvs_get_blob(h, NVS_KEY_STD,  s_std,  &std_sz)            == ESP_OK);

    nvs_close(h);

    if (ok) {
        s_n_feat = (int)n_feat;
        s_valid  = true;
        printf("[Calib] Loaded from NVS (%d features)\n", s_n_feat);
    } else {
        printf("[Calib] No valid calibration in NVS — identity transform active\n");
    }

    return ok;
}

void calibration_apply(float *feat) {
    if (!s_valid) return;
    for (int i = 0; i < s_n_feat; i++) {
        feat[i] = (feat[i] - s_mean[i]) / s_std[i];
    }
}

/**
 * @brief Reconcile the calibration scale with the model's training scale.
 *
 * The models expect features in (x - mu) / sigma_train space, where sigma_train
 * is the across-gesture std measured at training time. REST-only calibration
 * produces a std vector that is much smaller, so dividing by it inflates
 * calibrated features and saturates the LDA/MLP softmax (the dominant cause of
 * collapsed predictions). The REST mean is still used as the per-session offset
 * to track electrode-placement drift; only the scale is corrected here.
 *
 * If the header was exported with sigma_train (MODEL_HAS_TRAIN_STD), the exact
 * training std is used. Otherwise we fall back to flooring the REST std relative
 * to the across-feature mean so near-constant features can't be over-amplified.
 *
 * @param std    Per-feature std vector (modified in place).
 * @param n_feat Number of features.
 */
static void calibration_set_train_scale(float *std, int n_feat) {
    if (n_feat <= 0) return;

#if defined(MODEL_HAS_TRAIN_STD) && MODEL_HAS_TRAIN_STD
    int n = (n_feat < MODEL_NUM_FEATURES) ? n_feat : MODEL_NUM_FEATURES;
    for (int f = 0; f < n; f++) {
        if (MODEL_FEAT_STD[f] > 1e-6f) {
            std[f] = MODEL_FEAT_STD[f];
        }
    }
#else
    /* No baked training std yet — re-export model_weights.h (export_to_header)
     * to enable the exact path. Until then, floor the REST std at a fraction of
     * the across-feature mean std to bound feature inflation. */
    float sum = 0.0f;
    for (int f = 0; f < n_feat; f++) sum += std[f];
    float floor_val = 0.25f * (sum / n_feat);
    if (floor_val < 1e-6f) floor_val = 1e-6f;
    for (int f = 0; f < n_feat; f++) {
        if (std[f] < floor_val) std[f] = floor_val;
    }
#endif
}

bool calibration_update(const float *X_flat, int n_windows, int n_feat) {
    if (n_windows < 10 || n_feat <= 0 || n_feat > CALIB_MAX_FEATURES) {
        printf("[Calib] calibration_update: invalid args (%d windows, %d features)\n",
               n_windows, n_feat);
        return false;
    }

    s_n_feat = n_feat;

    /* Compute per-feature mean */
    memset(s_mean, 0, sizeof(s_mean));
    for (int w = 0; w < n_windows; w++) {
        for (int f = 0; f < n_feat; f++) {
            s_mean[f] += X_flat[w * n_feat + f];
        }
    }
    for (int f = 0; f < n_feat; f++) {
        s_mean[f] /= n_windows;
    }

    /* Compute per-feature std (with epsilon floor) */
    memset(s_std, 0, sizeof(s_std));
    for (int w = 0; w < n_windows; w++) {
        for (int f = 0; f < n_feat; f++) {
            float d = X_flat[w * n_feat + f] - s_mean[f];
            s_std[f] += d * d;
        }
    }
    for (int f = 0; f < n_feat; f++) {
        float var = s_std[f] / n_windows;
        s_std[f]  = (var > 1e-12f) ? sqrtf(var) : 1e-6f;
    }

    /* Scale calibrated features into the model's training space (prevents the
     * softmax-saturation failure mode). See calibration_set_train_scale(). */
    calibration_set_train_scale(s_std, n_feat);

    /* Persist to NVS */
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) {
        printf("[Calib] calibration_update: failed to open NVS\n");
        return false;
    }

    esp_err_t err = ESP_OK;
    err |= nvs_set_blob(h, NVS_KEY_MEAN, s_mean, sizeof(s_mean));
    err |= nvs_set_blob(h, NVS_KEY_STD,  s_std,  sizeof(s_std));
    err |= nvs_set_i32 (h, NVS_KEY_NFEAT, (int32_t)n_feat);
    err |= nvs_set_u8  (h, NVS_KEY_VALID, 1u);
    err |= nvs_commit(h);
    nvs_close(h);

    if (err != ESP_OK) {
        printf("[Calib] calibration_update: NVS write failed (err=0x%x)\n", err);
        return false;
    }

    s_valid = true;
    printf("[Calib] Updated: %d REST windows, %d features saved to NVS\n",
           n_windows, n_feat);
    return true;
}

void calibration_reset(void) {
    s_valid  = false;
    s_n_feat = 0;
    memset(s_mean, 0, sizeof(s_mean));
    memset(s_std,  0, sizeof(s_std));
}

bool calibration_is_valid(void) {
    return s_valid;
}
