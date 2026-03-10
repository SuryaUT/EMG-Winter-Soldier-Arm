/**
 * @file bicep.c
 * @brief Bicep channel subsystem — binary flex/unflex detector (Phase 1).
 */

#include "bicep.h"
#include "inference.h"   /* inference_get_bicep_rms() */
#include "nvs_flash.h"
#include "nvs.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Tuning constants */
#define BICEP_WINDOW_SAMPLES  50     /**< 50 ms window at 1 kHz */
#define BICEP_FLEX_MULTIPLIER 2.5f   /**< threshold = rest_rms × 2.5 */
#define BICEP_HYSTERESIS      1.3f   /**< scale factor to enter flex (prevents toggling) */

/* NVS storage */
#define BICEP_NVS_NAMESPACE "bicep_calib"
#define BICEP_NVS_KEY_THRESH "threshold"
#define BICEP_NVS_KEY_VALID  "calib_ok"

/* Module state */
static float         s_threshold_mv = 0.0f;
static bicep_state_t s_state        = BICEP_STATE_REST;

/*******************************************************************************
 * Public API
 ******************************************************************************/

float bicep_calibrate(const uint16_t *ch3_samples, int n_samples) {
    if (n_samples <= 0) return 0.0f;

    float rms_sq = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float v = (float)ch3_samples[i];
        rms_sq += v * v;
    }
    float rest_rms    = sqrtf(rms_sq / n_samples);
    s_threshold_mv    = rest_rms * BICEP_FLEX_MULTIPLIER;
    s_state           = BICEP_STATE_REST;

    printf("[Bicep] Calibrated: rest_rms=%.1f mV, threshold=%.1f mV\n",
           rest_rms, s_threshold_mv);

    bicep_save_threshold(s_threshold_mv);
    return s_threshold_mv;
}

bicep_state_t bicep_detect(void) {
    if (s_threshold_mv <= 0.0f) {
        return BICEP_STATE_REST;  /* Not calibrated */
    }

    float rms = inference_get_bicep_rms(BICEP_WINDOW_SAMPLES);

    /* Hysteretic threshold: need FLEX_MULTIPLIER × threshold to enter flex,
     * drop below threshold to return to rest. */
    if (s_state == BICEP_STATE_REST) {
        if (rms > s_threshold_mv * BICEP_HYSTERESIS) {
            s_state = BICEP_STATE_FLEX;
        }
    } else {  /* BICEP_STATE_FLEX */
        if (rms < s_threshold_mv) {
            s_state = BICEP_STATE_REST;
        }
    }

    return s_state;
}

bool bicep_save_threshold(float threshold_mv) {
    nvs_handle_t h;
    if (nvs_open(BICEP_NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) {
        printf("[Bicep] Failed to open NVS for write\n");
        return false;
    }

    esp_err_t err = ESP_OK;
    err |= nvs_set_blob(h, BICEP_NVS_KEY_THRESH, &threshold_mv, sizeof(threshold_mv));
    err |= nvs_set_u8  (h, BICEP_NVS_KEY_VALID, 1u);
    err |= nvs_commit(h);
    nvs_close(h);

    if (err != ESP_OK) {
        printf("[Bicep] NVS write failed (err=0x%x)\n", err);
        return false;
    }
    printf("[Bicep] Threshold %.1f mV saved to NVS\n", threshold_mv);
    return true;
}

bool bicep_load_threshold(float *threshold_mv_out) {
    nvs_handle_t h;
    if (nvs_open(BICEP_NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) {
        return false;
    }

    uint8_t valid = 0;
    float   thresh = 0.0f;
    size_t  sz = sizeof(thresh);

    bool ok = (nvs_get_u8  (h, BICEP_NVS_KEY_VALID,  &valid)          == ESP_OK) &&
              (valid == 1)                                                         &&
              (nvs_get_blob(h, BICEP_NVS_KEY_THRESH, &thresh, &sz)    == ESP_OK) &&
              (thresh > 0.0f);
    nvs_close(h);

    if (ok) {
        s_threshold_mv = thresh;
        if (threshold_mv_out) *threshold_mv_out = thresh;
        printf("[Bicep] Loaded threshold: %.1f mV\n", thresh);
    }
    return ok;
}

float bicep_calibrate_from_buffer(int n_samples) {
    float rest_rms = inference_get_bicep_rms(n_samples);
    if (rest_rms < 1e-6f) {
        printf("[Bicep] WARNING: rest RMS ≈ 0 — buffer may not be filled yet\n");
        return 0.0f;
    }

    s_threshold_mv = rest_rms * BICEP_FLEX_MULTIPLIER;
    s_state        = BICEP_STATE_REST;

    printf("[Bicep] Calibrated (filtered): rest_rms=%.2f, threshold=%.2f\n",
           rest_rms, s_threshold_mv);

    bicep_save_threshold(s_threshold_mv);
    return s_threshold_mv;
}

void bicep_set_threshold(float threshold_mv) {
    s_threshold_mv = threshold_mv;
    s_state = BICEP_STATE_REST;
}

float bicep_get_threshold(void) {
    return s_threshold_mv;
}
