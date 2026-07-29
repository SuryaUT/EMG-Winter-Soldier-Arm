/**
 * @file bicep.c
 * @brief Bicep channel subsystem — binary flex/unflex detector (Phase 1).
 */

#include "bicep.h"
#include "inference.h"       /* inference_get_bicep_rms() */
#include "drivers/hand.h"    /* hand_set_finger_angle() */
#include "nvs_flash.h"
#include "nvs.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Tuning constants */
#define BICEP_WINDOW_SAMPLES  50     /**< 50 ms window at 1 kHz */
#define BICEP_FLEX_MULTIPLIER 2.5f   /**< threshold = rest_rms × 2.5 */
#define BICEP_HYSTERESIS      1.3f   /**< scale factor to enter flex (prevents toggling) */

/* Servo actuation angles (JOINT_BICEP -> PCA channel 6) */
#define BICEP_FLEX_ANGLE      140.0f /**< max flex angle (degrees) */
#define BICEP_REST_ANGLE       60.0f /**< min / unflexed angle (degrees) */

/* Proportional-control tuning */
#define BICEP_DEADZONE        0.05f  /**< levels below this snap to 0 (anti-jitter at rest) */
#define BICEP_LEVEL_EMA       0.80f  /**< smoothing on activation level: fraction of previous kept */
#define BICEP_ANGLE_RATE_DEG  6.0f   /**< max servo angle change per hop (deg) — caps slew speed */
#define BICEP_ANGLE_WRITE_EPS 0.5f   /**< skip the I2C write if the angle moved less than this */

/* NVS storage */
#define BICEP_NVS_NAMESPACE "bicep_calib"
#define BICEP_NVS_KEY_THRESH "threshold"
#define BICEP_NVS_KEY_VALID  "calib_ok"
#define BICEP_NVS_KEY_REST   "rest_rms"
#define BICEP_NVS_KEY_MAX    "max_rms"
#define BICEP_NVS_KEY_PROP   "prop_ok"

/* Module state (binary detector) */
static float         s_threshold_mv = 0.0f;
static bicep_state_t s_state        = BICEP_STATE_REST;

/* Module state (proportional control) */
static float s_rest_rms     = 0.0f;              /**< RMS floor (level 0.0) */
static float s_max_rms      = 0.0f;              /**< RMS ceiling (level 1.0) */
static float s_level_ema    = 0.0f;              /**< smoothed activation level [0,1] */
static float s_angle_cur    = BICEP_REST_ANGLE;  /**< rate-limited servo angle */
static float s_angle_written = -1.0f;            /**< last angle actually sent to servo */

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

void bicep_apply(bicep_state_t state) {
    static int last_applied = -1;             /* forces a command on first call */
    if ((int)state == last_applied) return;   /* edge-triggered: only on change */
    last_applied = (int)state;

    float angle = (state == BICEP_STATE_FLEX) ? BICEP_FLEX_ANGLE : BICEP_REST_ANGLE;
    hand_set_finger_angle(JOINT_BICEP, angle);
    printf("[Bicep] %s -> %.0f deg\n",
           (state == BICEP_STATE_FLEX) ? "FLEX" : "REST", angle);
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

/*******************************************************************************
 * Proportional control
 ******************************************************************************/

float bicep_current_rms(void) {
    return inference_get_bicep_rms(BICEP_WINDOW_SAMPLES);
}

void bicep_set_proportional(float rest_rms, float max_rms) {
    s_rest_rms  = rest_rms;
    s_max_rms   = max_rms;
    s_level_ema = 0.0f;
    s_angle_cur = BICEP_REST_ANGLE;
}

bool bicep_save_proportional(float rest_rms, float max_rms) {
    nvs_handle_t h;
    if (nvs_open(BICEP_NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) {
        printf("[Bicep] Failed to open NVS for write\n");
        return false;
    }

    esp_err_t err = ESP_OK;
    err |= nvs_set_blob(h, BICEP_NVS_KEY_REST, &rest_rms, sizeof(rest_rms));
    err |= nvs_set_blob(h, BICEP_NVS_KEY_MAX,  &max_rms,  sizeof(max_rms));
    err |= nvs_set_u8  (h, BICEP_NVS_KEY_PROP, 1u);
    err |= nvs_commit(h);
    nvs_close(h);

    if (err != ESP_OK) {
        printf("[Bicep] NVS proportional write failed (err=0x%x)\n", err);
        return false;
    }
    printf("[Bicep] Proportional calib saved: rest=%.2f max=%.2f\n",
           rest_rms, max_rms);
    return true;
}

bool bicep_load_proportional(float *rest_rms_out, float *max_rms_out) {
    nvs_handle_t h;
    if (nvs_open(BICEP_NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) {
        return false;
    }

    uint8_t valid = 0;
    float   rest = 0.0f, max = 0.0f;
    size_t  sz_rest = sizeof(rest), sz_max = sizeof(max);

    bool ok = (nvs_get_u8  (h, BICEP_NVS_KEY_PROP, &valid)           == ESP_OK) &&
              (valid == 1)                                                        &&
              (nvs_get_blob(h, BICEP_NVS_KEY_REST, &rest, &sz_rest)  == ESP_OK) &&
              (nvs_get_blob(h, BICEP_NVS_KEY_MAX,  &max,  &sz_max)   == ESP_OK) &&
              (max > rest);
    nvs_close(h);

    if (ok) {
        bicep_set_proportional(rest, max);
        if (rest_rms_out) *rest_rms_out = rest;
        if (max_rms_out)  *max_rms_out  = max;
        printf("[Bicep] Loaded proportional calib: rest=%.2f max=%.2f\n",
               rest, max);
    }
    return ok;
}

float bicep_get_level(void) {
    if (s_max_rms <= s_rest_rms) {
        return 0.0f;  /* not calibrated / degenerate range */
    }

    float rms   = inference_get_bicep_rms(BICEP_WINDOW_SAMPLES);
    float level = (rms - s_rest_rms) / (s_max_rms - s_rest_rms);

    if (level < 0.0f) level = 0.0f;
    if (level > 1.0f) level = 1.0f;
    if (level < BICEP_DEADZONE) level = 0.0f;  /* relaxed muscle reads exactly 0 */
    return level;
}

void bicep_apply_proportional(void) {
    float level = bicep_get_level();

    /* Smooth the raw level — EMG RMS is jittery and would make the servo buzz. */
    s_level_ema = BICEP_LEVEL_EMA * s_level_ema +
                  (1.0f - BICEP_LEVEL_EMA) * level;

    float target = BICEP_REST_ANGLE +
                   (BICEP_FLEX_ANGLE - BICEP_REST_ANGLE) * s_level_ema;

    /* Rate-limit so the servo slews smoothly instead of snapping. */
    float delta = target - s_angle_cur;
    if (delta >  BICEP_ANGLE_RATE_DEG) delta =  BICEP_ANGLE_RATE_DEG;
    if (delta < -BICEP_ANGLE_RATE_DEG) delta = -BICEP_ANGLE_RATE_DEG;
    s_angle_cur += delta;

    /* Hard safety clamp: the math above already keeps us in range (level is
     * clamped to [0,1]), but guard the servo command regardless of what any
     * future upstream change feeds in. */
    if (s_angle_cur < BICEP_REST_ANGLE) s_angle_cur = BICEP_REST_ANGLE;
    if (s_angle_cur > BICEP_FLEX_ANGLE) s_angle_cur = BICEP_FLEX_ANGLE;

    /* Only touch the bus when the pose actually moved — holding a level costs
     * nothing, unlike commanding the same angle every 25 ms hop. */
    if (fabsf(s_angle_cur - s_angle_written) >= BICEP_ANGLE_WRITE_EPS) {
        hand_set_finger_angle(JOINT_BICEP, s_angle_cur);
        s_angle_written = s_angle_cur;
    }
}
