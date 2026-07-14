/**
 * @file inference.c
 * @brief Implementation of EMG inference engine.
 */

#include "inference.h"
#include "calibration.h"
#include "config/config.h"
#include "model_weights.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "dsps_dotprod.h"

#if MODEL_EXPAND_FEATURES
#include "dsps_fft2r.h"   /* esp-dsp: complex 2-radix FFT */
#define FFT_N 256         /* Must match Python fft_n=256 */
#endif

// --- Change B: IIR Bandpass Filter (20–450 Hz, 2nd-order Butterworth @ 1 kHz) ---
// Two cascaded biquad sections, Direct Form II Transposed.
// Computed via scipy.signal.butter(2, [20,450], btype='bandpass', fs=1000, output='sos').
// b coefficients [b0, b1, b2] per section:
#define IIR_NUM_SECTIONS 2
static const float IIR_B[IIR_NUM_SECTIONS][3] = {
    { 0.7320224766f,  1.4640449531f,  0.7320224766f },  /* section 0 */
    { 1.0000000000f, -2.0000000000f,  1.0000000000f },  /* section 1 */
};
// Feedback coefficients [a1, a2] per section (a0 = 1, implicit):
static const float IIR_A[IIR_NUM_SECTIONS][2] = {
    {  1.5597081442f,  0.6416146818f },  /* section 0 */
    { -1.8224796027f,  0.8372542588f },  /* section 1 */
};

// --- State ---
static float    window_buffer[INFERENCE_WINDOW_SIZE][NUM_CHANNELS];
static float    biquad_w[IIR_NUM_SECTIONS][NUM_CHANNELS][2];  /* biquad delay states */

#if MODEL_EXPAND_FEATURES
static bool     s_fft_ready = false;
#endif
static int buffer_head = 0;
static int samples_collected = 0;

void inference_init(void) {
  memset(window_buffer, 0, sizeof(window_buffer));
  memset(biquad_w,      0, sizeof(biquad_w));
  buffer_head = 0;
  samples_collected = 0;

#if MODEL_EXPAND_FEATURES
  if (!s_fft_ready) {
    dsps_fft2r_init_fc32(NULL, FFT_N);
    s_fft_ready = true;
  }
#endif
}

bool inference_add_sample(uint16_t *channels) {
  // Convert to float, apply per-channel biquad bandpass, then store in circular buffer.
  for (int i = 0; i < NUM_CHANNELS; i++) {
    float x = (float)channels[i];
    // Cascade IIR_NUM_SECTIONS biquad sections (Direct Form II Transposed)
    for (int s = 0; s < IIR_NUM_SECTIONS; s++) {
      float y          = IIR_B[s][0] * x + biquad_w[s][i][0];
      biquad_w[s][i][0] = IIR_B[s][1] * x - IIR_A[s][0] * y + biquad_w[s][i][1];
      biquad_w[s][i][1] = IIR_B[s][2] * x - IIR_A[s][1] * y;
      x = y;
    }
    window_buffer[buffer_head][i] = x;
  }

  buffer_head = (buffer_head + 1) % INFERENCE_WINDOW_SIZE;

  if (samples_collected < INFERENCE_WINDOW_SIZE) {
    samples_collected++;
    return false;
  }

  return true; // Buffer is full (always ready in sliding window, but caller
               // controls stride)
}

// --- Feature Extraction ---

/* ── helpers used by compute_features_expanded ──────────────────────────── */

#if MODEL_EXPAND_FEATURES

/** Solve 4×4 linear system A·x = b via Gaussian elimination with partial pivoting.
 *  Returns false and leaves x zeroed if the matrix is singular. */
static bool solve_4x4(float A[4][4], const float b[4], float x[4]) {
  float M[4][5];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) M[i][j] = A[i][j];
    M[i][4] = b[i];
  }
  for (int col = 0; col < 4; col++) {
    int pivot = col;
    float maxv = fabsf(M[col][col]);
    for (int row = col + 1; row < 4; row++) {
      if (fabsf(M[row][col]) > maxv) { maxv = fabsf(M[row][col]); pivot = row; }
    }
    if (maxv < 1e-8f) { x[0]=x[1]=x[2]=x[3]=0.f; return false; }
    if (pivot != col) {
      float tmp[5];
      memcpy(tmp, M[col], 5*sizeof(float));
      memcpy(M[col], M[pivot], 5*sizeof(float));
      memcpy(M[pivot], tmp, 5*sizeof(float));
    }
    for (int row = col + 1; row < 4; row++) {
      float f = M[row][col] / M[col][col];
      for (int j = col; j < 5; j++) M[row][j] -= f * M[col][j];
    }
  }
  for (int row = 3; row >= 0; row--) {
    x[row] = M[row][4];
    for (int j = row + 1; j < 4; j++) x[row] -= M[row][j] * x[j];
    x[row] /= M[row][row];
  }
  return true;
}

/**
 * @brief Full 69-feature extraction (Change 1).
 *
 * Per channel (20): RMS, WL, ZC, SSC, MAV, VAR, IEMG, WAMP,
 *                   AR1-AR4, MNF, MDF, PKF, MNP, BP0-BP3
 * Cross-channel (9 for 3 channels): corr, log-RMS-ratio, cov for each pair
 *
 * Feature layout must exactly match EMGFeatureExtractor._EXPANDED_KEYS +
 * cross-channel order in the Python class.
 */
static void compute_features_expanded(float *features_out) {
  memset(features_out, 0, MODEL_NUM_FEATURES * sizeof(float));

  /* Persistent buffers for centered signals (3 ch × 150 samples) */
  static float ch_signals[HAND_NUM_CHANNELS][INFERENCE_WINDOW_SIZE];
  /* 16-byte alignment is REQUIRED: on the ESP32-S3 dsps_fft2r_fc32 dispatches to
   * the AES3 SIMD kernel, which silently produces a scrambled spectrum on an
   * unaligned buffer. */
  static float s_fft_buf[FFT_N * 2] __attribute__((aligned(16)));  /* Complex interleaved [re,im,...] */

  float ch_rms[HAND_NUM_CHANNELS];
  float norm_sq = 0.0f;

  /* ──────────────────────────────────────────────────────────────────────
   * Pass 1: per-channel TD + AR + spectral features
   * ────────────────────────────────────────────────────────────────────── */
  for (int ch = 0; ch < HAND_NUM_CHANNELS; ch++) {
    /* Read channel from circular buffer (oldest-first) */
    float *sig = ch_signals[ch];
    float sum = 0.0f;
    int idx = buffer_head;
    for (int i = 0; i < INFERENCE_WINDOW_SIZE; i++) {
      sig[i] = window_buffer[idx][ch];
      sum += sig[i];
      idx = (idx + 1) % INFERENCE_WINDOW_SIZE;
    }
    /* DC removal */
    float mean = sum / INFERENCE_WINDOW_SIZE;
    float sq_sum = 0.0f;
    for (int i = 0; i < INFERENCE_WINDOW_SIZE; i++) {
      sig[i] -= mean;
      sq_sum += sig[i] * sig[i];
    }
    float sig_mean = 0.0f;  /* mean variance is taken about; exactly 0 after DC removal */
    /* Change 4 — Reinhard tone-mapping: 64·x / (32 + |x|) */
#if MODEL_USE_REINHARD
    sq_sum = 0.0f;
    float sig_sum = 0.0f;
    for (int i = 0; i < INFERENCE_WINDOW_SIZE; i++) {
      float x = sig[i];
      sig[i] = 64.0f * x / (32.0f + fabsf(x));
      sq_sum  += sig[i] * sig[i];
      sig_sum += sig[i];
    }
    sig_mean = sig_sum / INFERENCE_WINDOW_SIZE;  /* Reinhard leaves a small nonzero mean */
#endif

    float rms = sqrtf(sq_sum / INFERENCE_WINDOW_SIZE);
    ch_rms[ch] = rms;
    norm_sq   += rms * rms;

    float zc_thresh  = FEAT_ZC_THRESH  * rms;
    float ssc_thresh = (FEAT_SSC_THRESH * rms) * (FEAT_SSC_THRESH * rms);

    /* TD features */
    float wl = 0.0f, mav = 0.0f, iemg = 0.0f;
    int   zc = 0, ssc = 0, wamp = 0;
    for (int i = 0; i < INFERENCE_WINDOW_SIZE; i++) {
      float a = fabsf(sig[i]);
      mav  += a;
      iemg += a;
    }
    mav /= INFERENCE_WINDOW_SIZE;
    /* True variance — matches Python np.var (mean-square minus mean²). */
    float var_val = sq_sum / INFERENCE_WINDOW_SIZE - sig_mean * sig_mean;

    for (int i = 0; i < INFERENCE_WINDOW_SIZE - 1; i++) {
      float diff  = sig[i+1] - sig[i];
      float adiff = fabsf(diff);
      wl += adiff;
      if (adiff > zc_thresh) wamp++;
      if ((sig[i] > 0.0f && sig[i+1] < 0.0f) ||
          (sig[i] < 0.0f && sig[i+1] > 0.0f)) {
        if (adiff > zc_thresh) zc++;
      }
      if (i < INFERENCE_WINDOW_SIZE - 2) {
        float d1 = sig[i+1] - sig[i];
        float d2 = sig[i+1] - sig[i+2];
        if ((d1 * d2) > ssc_thresh) ssc++;
      }
    }

    /* AR(4) via Yule-Walker */
    float r[5] = {0};
    for (int lag = 0; lag < 5; lag++) {
      for (int i = 0; i < INFERENCE_WINDOW_SIZE - lag; i++)
        r[lag] += sig[i] * sig[i + lag];
      r[lag] /= INFERENCE_WINDOW_SIZE;
    }
    float T[4][4], b_ar[4], ar[4] = {0};
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) T[i][j] = r[abs(i - j)];
      b_ar[i] = r[i + 1];
    }
    solve_4x4(T, b_ar, ar);

    /* Spectral features via FFT (zero-pad to FFT_N) */
    memset(s_fft_buf, 0, sizeof(s_fft_buf));
    for (int i = 0; i < INFERENCE_WINDOW_SIZE; i++) {
      s_fft_buf[2*i]     = sig[i];
      s_fft_buf[2*i + 1] = 0.0f;
    }
    dsps_fft2r_fc32(s_fft_buf, FFT_N);
    dsps_bit_rev_fc32(s_fft_buf, FFT_N);

    float total_power = 1e-10f;
    const float freq_step = (float)EMG_SAMPLE_RATE_HZ / FFT_N;
    /* numpy rfft(signal, n=FFT_N) returns FFT_N/2 + 1 bins (DC..Nyquist) — match
     * the bin count exactly so total power, MNF, MDF, MNP and band powers agree. */
    float pwr[FFT_N / 2 + 1];
    for (int k = 0; k <= FFT_N / 2; k++) {
      float re = s_fft_buf[2*k], im = s_fft_buf[2*k + 1];
      pwr[k] = re*re + im*im;
      total_power += pwr[k];
    }

    float mnf = 0.0f;
    for (int k = 0; k <= FFT_N / 2; k++) mnf += k * freq_step * pwr[k];
    mnf /= total_power;

    float cumsum = 0.0f, half = total_power / 2.0f;
    int mdf_k = FFT_N / 2;
    for (int k = 0; k <= FFT_N / 2; k++) {
      cumsum += pwr[k];
      if (cumsum >= half) { mdf_k = k; break; }
    }
    float mdf = mdf_k * freq_step;

    int pkf_k = 0;
    for (int k = 1; k <= FFT_N / 2; k++) {
      if (pwr[k] > pwr[pkf_k]) pkf_k = k;
    }
    float pkf = pkf_k * freq_step;
    float mnp = total_power / (FFT_N / 2 + 1);

    float bp0 = 0, bp1 = 0, bp2 = 0, bp3 = 0;
    for (int k = 0; k <= FFT_N / 2; k++) {
      float f = k * freq_step;
      if      (f >= 20.f  && f < 80.f ) bp0 += pwr[k];
      else if (f >= 80.f  && f < 150.f) bp1 += pwr[k];
      else if (f >= 150.f && f < 250.f) bp2 += pwr[k];
      else if (f >= 250.f && f < 450.f) bp3 += pwr[k];
    }
    bp0 /= total_power; bp1 /= total_power;
    bp2 /= total_power; bp3 /= total_power;

    /* Store 20 features for this channel */
    int base = ch * 20;
    features_out[base +  0] = rms;
    features_out[base +  1] = wl;
    features_out[base +  2] = (float)zc;
    features_out[base +  3] = (float)ssc;
    features_out[base +  4] = mav;
    features_out[base +  5] = var_val;
    features_out[base +  6] = iemg;
    features_out[base +  7] = (float)wamp;
    features_out[base +  8] = ar[0];
    features_out[base +  9] = ar[1];
    features_out[base + 10] = ar[2];
    features_out[base + 11] = ar[3];
    features_out[base + 12] = mnf;
    features_out[base + 13] = mdf;
    features_out[base + 14] = pkf;
    features_out[base + 15] = mnp;
    features_out[base + 16] = bp0;
    features_out[base + 17] = bp1;
    features_out[base + 18] = bp2;
    features_out[base + 19] = bp3;
  }

  /* ──────────────────────────────────────────────────────────────────────
   * Amplitude normalization (matches Python normalize=True behaviour)
   * ────────────────────────────────────────────────────────────────────── */
  float norm_factor = sqrtf(norm_sq);
  if (norm_factor < 1e-6f) norm_factor = 1e-6f;
  for (int ch = 0; ch < HAND_NUM_CHANNELS; ch++) {
    int base = ch * 20;
    features_out[base + 0] /= norm_factor;  /* rms  */
    features_out[base + 1] /= norm_factor;  /* wl   */
    features_out[base + 4] /= norm_factor;  /* mav  */
    features_out[base + 6] /= norm_factor;  /* iemg */
  }

  /* ──────────────────────────────────────────────────────────────────────
   * Cross-channel features (3 pairs × 3 features = 9)
   * Order: (0,1), (0,2), (1,2)  →  corr, lrms, cov
   * ────────────────────────────────────────────────────────────────────── */
  int xc_base = HAND_NUM_CHANNELS * 20;  /* = 60 */
  int xc_idx  = 0;
  for (int i = 0; i < HAND_NUM_CHANNELS; i++) {
    for (int j = i + 1; j < HAND_NUM_CHANNELS; j++) {
      float ri = ch_rms[i] + 1e-10f;
      float rj = ch_rms[j] + 1e-10f;

      float dot_ij = 0.0f;
      for (int k = 0; k < INFERENCE_WINDOW_SIZE; k++)
        dot_ij += ch_signals[i][k] * ch_signals[j][k];
      float n_inv = 1.0f / INFERENCE_WINDOW_SIZE;

      float corr = dot_ij * n_inv / (ri * rj);
      if (corr >  1.0f) corr =  1.0f;
      if (corr < -1.0f) corr = -1.0f;

      float lrms = logf(ri / rj);
      float cov  = dot_ij * n_inv / (norm_factor * norm_factor);

      features_out[xc_base + xc_idx * 3 + 0] = corr;
      features_out[xc_base + xc_idx * 3 + 1] = lrms;
      features_out[xc_base + xc_idx * 3 + 2] = cov;
      xc_idx++;
    }
  }
}

#endif  /* MODEL_EXPAND_FEATURES */

// --- Feature extraction (public wrapper used by inference_ensemble.c) ---

void inference_extract_features(float *features_out) {
#if !MODEL_EXPAND_FEATURES
#error "MODEL_EXPAND_FEATURES=0 is no longer supported: the legacy 4-feature/channel \
extractor was removed. The models, the ensemble gathers and the 69-feature firmware \
layout all assume the expanded set. Re-export model_weights.h from Python."
#endif
  compute_features_expanded(features_out);
  calibration_apply(features_out);
}

// --- Raw LDA probability (for multi-model voting) ---

void inference_predict_raw(const float *features, float *proba_out) {
  float raw_scores[MODEL_NUM_CLASSES];
  float max_score = -1e9f;

  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    float dot;
    dsps_dotprod_f32(features, LDA_WEIGHTS[c], &dot, MODEL_NUM_FEATURES);
    raw_scores[c] = dot + LDA_INTERCEPTS[c];
    if (raw_scores[c] > max_score) max_score = raw_scores[c];
  }

  float sum_exp = 0.0f;
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    proba_out[c] = expf(raw_scores[c] - max_score);
    sum_exp += proba_out[c];
  }
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    proba_out[c] /= sum_exp;
  }
}

const char *inference_get_class_name(int class_idx) {
  if (class_idx >= 0 && class_idx < MODEL_NUM_CLASSES) {
    return MODEL_CLASS_NAMES[class_idx];
  }
  return "UNKNOWN";
}

int inference_get_gesture_enum(int class_idx) {
  const char *name = inference_get_class_name(class_idx);
  return inference_get_gesture_by_name(name);
}

int inference_get_gesture_by_name(const char *name) {
  // Accepts both lowercase (Python output) and uppercase (C enum name style).
  // Add a new case here whenever a gesture is added to gesture_t in config.h.
  if (strcmp(name, "rest") == 0 || strcmp(name, "REST") == 0)
    return GESTURE_REST;
  if (strcmp(name, "fist") == 0 || strcmp(name, "FIST") == 0)
    return GESTURE_FIST;
  if (strcmp(name, "open") == 0 || strcmp(name, "OPEN") == 0)
    return GESTURE_OPEN;
  if (strcmp(name, "hook_em") == 0 || strcmp(name, "HOOK_EM") == 0)
    return GESTURE_HOOK_EM;
  if (strcmp(name, "thumbs_up") == 0 || strcmp(name, "THUMBS_UP") == 0)
    return GESTURE_THUMBS_UP;
  return GESTURE_NONE;
}

float inference_get_bicep_rms(int n_samples) {
  if (samples_collected < INFERENCE_WINDOW_SIZE) return 0.0f;
  if (n_samples > INFERENCE_WINDOW_SIZE) n_samples = INFERENCE_WINDOW_SIZE;

  float sum_sq = 0.0f;
  // Walk backwards from buffer_head (oldest = buffer_head, newest = buffer_head - 1)
  int start = (buffer_head - n_samples + INFERENCE_WINDOW_SIZE) % INFERENCE_WINDOW_SIZE;
  int idx = start;
  for (int i = 0; i < n_samples; i++) {
    float v = window_buffer[idx][3];  // channel 3 = bicep
    sum_sq += v * v;
    idx = (idx + 1) % INFERENCE_WINDOW_SIZE;
  }
  return sqrtf(sum_sq / n_samples);
}
