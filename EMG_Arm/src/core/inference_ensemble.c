/**
 * @file inference_ensemble.c
 * @brief 3-specialist-LDA + meta-LDA ensemble inference pipeline (Change F).
 *
 * Guarded by MODEL_USE_ENSEMBLE in model_weights.h.
 * When 0, provides empty stubs so the file compiles unconditionally.
 */

#include "inference_ensemble.h"
#include "inference.h"
#include "model_weights.h"
#include "dsps_dotprod.h"

#if MODEL_USE_ENSEMBLE

#include "inference_mlp.h"
#include "model_weights_ensemble.h"
#include "calibration.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#define ENSEMBLE_EMA_ALPHA      0.70f
#define ENSEMBLE_CONF_THRESHOLD 0.50f  /**< below this: escalate to MLP fallback */
#define REJECT_THRESHOLD        0.40f  /**< below this even after MLP: hold output */
#define REST_ACTIVITY_THRESHOLD 0.05f  /**< total RMS gate — skip inference during rest */

/* Class index for "rest" — looked up once in init so we don't return
 * the gesture_t enum value (GESTURE_REST = 1) which would be misinterpreted
 * as class index 1 ("hook_em").  See Bug 2 in the code review. */
static int s_rest_class_idx = 0;

/* EMA probability state */
static float s_smoothed[MODEL_NUM_CLASSES];

/* Majority vote ring buffer (window = 5) */
static int s_vote_history[5];
static int s_vote_head = 0;

/* Debounce state */
static int s_current_output = -1;
static int s_pending_output = -1;
static int s_pending_count  = 0;

/* ── Generic LDA softmax ──────────────────────────────────────────────────── */

/**
 * Compute softmax class probabilities from a flat feature vector.
 *
 * @param feat         Feature vector (contiguous, length n_feat).
 * @param n_feat       Number of features.
 * @param weights_flat Row-major weight matrix, shape [n_classes][n_feat].
 * @param intercepts   Intercept vector, length n_classes.
 * @param n_classes    Number of output classes.
 * @param proba_out    Output probabilities, length n_classes (caller-allocated).
 */
static void lda_softmax(const float *feat, int n_feat,
                         const float *weights_flat, const float *intercepts,
                         int n_classes, float *proba_out) {
    float raw[MODEL_NUM_CLASSES];
    float max_raw = -1e9f;
    float sum_exp = 0.0f;

    for (int c = 0; c < n_classes; c++) {
        float dot;
        const float *w = weights_flat + c * n_feat;
        dsps_dotprod_f32(feat, w, &dot, n_feat);
        raw[c] = dot + intercepts[c];
        if (raw[c] > max_raw) max_raw = raw[c];
    }
    for (int c = 0; c < n_classes; c++) {
        proba_out[c] = expf(raw[c] - max_raw);
        sum_exp += proba_out[c];
    }
    for (int c = 0; c < n_classes; c++) {
        proba_out[c] /= sum_exp;
    }
}

/* ── Public API ───────────────────────────────────────────────────────────── */

void inference_ensemble_init(void) {
    /* Find the class index for "rest" once, so we can return it correctly
     * from the activity gate without confusing class indices with gesture_t. */
    s_rest_class_idx = 0;
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        if (strcmp(MODEL_CLASS_NAMES[i], "rest") == 0) {
            s_rest_class_idx = i;
            break;
        }
    }

    for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
        s_smoothed[c] = 1.0f / MODEL_NUM_CLASSES;
    }
    for (int i = 0; i < 5; i++) {
        s_vote_history[i] = -1;
    }
    s_vote_head      = 0;
    s_current_output = -1;
    s_pending_output = -1;
    s_pending_count  = 0;
}

void inference_ensemble_predict_raw(const float *features, float *proba_out) {
    /* Gather TD features (non-contiguous: 12 per channel × 3 channels) */
    float td_buf[TD_NUM_FEATURES];
    for (int ch = 0; ch < HAND_NUM_CHANNELS; ch++) {
        memcpy(td_buf + ch * 12,
               features + ch * ENSEMBLE_PER_CH_FEATURES,
               12 * sizeof(float));
    }

    /* Gather FD features (non-contiguous: 8 per channel × 3 channels) */
    float fd_buf[FD_NUM_FEATURES];
    for (int ch = 0; ch < HAND_NUM_CHANNELS; ch++) {
        memcpy(fd_buf + ch * 8,
               features + ch * ENSEMBLE_PER_CH_FEATURES + 12,
               8 * sizeof(float));
    }

    /* CC features are already contiguous at the end (indices 60-68) */
    const float *cc_buf = features + CC_FEAT_OFFSET;

    /* Specialist LDA predictions */
    float prob_td[MODEL_NUM_CLASSES];
    float prob_fd[MODEL_NUM_CLASSES];
    float prob_cc[MODEL_NUM_CLASSES];

    lda_softmax(td_buf, TD_NUM_FEATURES,
                (const float *)LDA_TD_WEIGHTS, LDA_TD_INTERCEPTS,
                MODEL_NUM_CLASSES, prob_td);
    lda_softmax(fd_buf, FD_NUM_FEATURES,
                (const float *)LDA_FD_WEIGHTS, LDA_FD_INTERCEPTS,
                MODEL_NUM_CLASSES, prob_fd);
    lda_softmax(cc_buf, CC_NUM_FEATURES,
                (const float *)LDA_CC_WEIGHTS, LDA_CC_INTERCEPTS,
                MODEL_NUM_CLASSES, prob_cc);

    /* Meta-LDA stacker */
    float meta_in[META_NUM_INPUTS];
    memcpy(meta_in,                       prob_td, MODEL_NUM_CLASSES * sizeof(float));
    memcpy(meta_in +   MODEL_NUM_CLASSES, prob_fd, MODEL_NUM_CLASSES * sizeof(float));
    memcpy(meta_in + 2*MODEL_NUM_CLASSES, prob_cc, MODEL_NUM_CLASSES * sizeof(float));

    lda_softmax(meta_in, META_NUM_INPUTS,
                (const float *)META_LDA_WEIGHTS, META_LDA_INTERCEPTS,
                MODEL_NUM_CLASSES, proba_out);
}

int inference_ensemble_predict(float *confidence) {
    /* 1. Extract + calibrate features (shared with single-model path) */
    float features[MODEL_NUM_FEATURES];
    inference_extract_features(features);   /* includes calibration_apply() */

    /* 2. Activity gate — skip inference during obvious REST */
    float total_rms_sq = 0.0f;
    for (int ch = 0; ch < HAND_NUM_CHANNELS; ch++) {
        /* RMS is the first feature per channel (index 0 in each 20-feature block) */
        float r = features[ch * ENSEMBLE_PER_CH_FEATURES];
        total_rms_sq += r * r;
    }
    if (sqrtf(total_rms_sq) < REST_ACTIVITY_THRESHOLD) {
        *confidence = 1.0f;
        return s_rest_class_idx;
    }

    /* 3. Run ensemble pipeline (raw probabilities) */
    float meta_probs[MODEL_NUM_CLASSES];
    inference_ensemble_predict_raw(features, meta_probs);

    /* 7. EMA smoothing on meta output */
    float max_smooth = 0.0f;
    int winner = 0;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
        s_smoothed[c] = ENSEMBLE_EMA_ALPHA * s_smoothed[c] +
                        (1.0f - ENSEMBLE_EMA_ALPHA) * meta_probs[c];
        if (s_smoothed[c] > max_smooth) {
            max_smooth = s_smoothed[c];
            winner = c;
        }
    }

    /* 8. Confidence cascade: escalate to MLP if meta-LDA is uncertain */
    if (max_smooth < ENSEMBLE_CONF_THRESHOLD) {
        float mlp_conf = 0.0f;
        int mlp_winner = inference_mlp_predict(features, MODEL_NUM_FEATURES, &mlp_conf);
        if (mlp_conf > max_smooth) {
            winner    = mlp_winner;
            max_smooth = mlp_conf;
        }
    }

    /* 9. Reject if still uncertain — hold current output */
    if (max_smooth < REJECT_THRESHOLD) {
        *confidence = max_smooth;
        return s_current_output >= 0 ? s_current_output : s_rest_class_idx;
    }

    *confidence = max_smooth;

    /* 10. Majority vote (window = 5) */
    s_vote_history[s_vote_head] = winner;
    s_vote_head = (s_vote_head + 1) % 5;

    int counts[MODEL_NUM_CLASSES];
    memset(counts, 0, sizeof(counts));
    for (int i = 0; i < 5; i++) {
        if (s_vote_history[i] >= 0) {
            counts[s_vote_history[i]]++;
        }
    }
    int majority = 0, majority_cnt = 0;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
        if (counts[c] > majority_cnt) {
            majority_cnt = counts[c];
            majority = c;
        }
    }

    /* 11. Debounce — need 3 consecutive predictions to change output */
    int final_out = (s_current_output >= 0) ? s_current_output : majority;

    if (s_current_output < 0) {
        /* First prediction ever */
        s_current_output = majority;
        final_out = majority;
    } else if (majority == s_current_output) {
        /* Staying in current gesture — reset pending */
        s_pending_output = majority;
        s_pending_count  = 1;
    } else if (majority == s_pending_output) {
        /* Same new gesture again */
        if (++s_pending_count >= 3) {
            s_current_output = majority;
            final_out = majority;
        }
    } else {
        /* New gesture candidate — start counting */
        s_pending_output = majority;
        s_pending_count  = 1;
    }

    return final_out;
}

#else  /* MODEL_USE_ENSEMBLE == 0 — compile-time stubs */

void inference_ensemble_init(void) {}

void inference_ensemble_predict_raw(const float *features, float *proba_out) {
    (void)features;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
        proba_out[c] = 1.0f / MODEL_NUM_CLASSES;
}

int inference_ensemble_predict(float *confidence) {
    if (confidence) *confidence = 0.0f;
    return 0;
}

#endif  /* MODEL_USE_ENSEMBLE */
