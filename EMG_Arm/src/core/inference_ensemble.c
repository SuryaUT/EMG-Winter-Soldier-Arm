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
    /* The ensemble is stateless: inference_ensemble_predict_raw() is a pure
     * function of the feature vector. Smoothing/voting live in main.c's
     * vote_postprocess(). Kept as a no-op so callers need not change. */
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

#else  /* MODEL_USE_ENSEMBLE == 0 — compile-time stubs */

void inference_ensemble_init(void) {}

void inference_ensemble_predict_raw(const float *features, float *proba_out) {
    (void)features;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
        proba_out[c] = 1.0f / MODEL_NUM_CLASSES;
}

#endif  /* MODEL_USE_ENSEMBLE */
