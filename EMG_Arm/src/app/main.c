/**
 * @file main.c
 * @brief Application entry point for the EMG-controlled robotic hand.
 *
 * Implements a robust handshake protocol with the host computer:
 * 1. Wait for "connect" command
 * 2. Acknowledge connection
 * 3. Wait for "start" command
 * 4. Stream EMG data
 * 5. Handle "stop" and "disconnect" commands
 *
 * @note This is Layer 4 (Application).
 */

#include "esp_timer.h"
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>
#include <stdio.h>
#include <string.h>

#include "config/config.h"
#include "core/bicep.h"
#include "core/calibration.h"
#include "core/gestures.h"
#include "core/inference.h"
#include "core/inference_ensemble.h"
#include "core/inference_mlp.h"
#include "core/model_weights.h"
#include "drivers/emg_sensor.h"
#include "drivers/hand.h"

#if MAIN_MODE == PARITY_DUMP
#if LIVE_EMG
#error "PARITY_DUMP requires LIVE_EMG 0 (replay data) in drivers/emg_sensor.h"
#endif
#include "drivers/replay_data.h"
#include <stdlib.h>
#endif

/*******************************************************************************
 * Constants
 ******************************************************************************/

#define CMD_BUFFER_SIZE 128
#define JSON_RESPONSE_SIZE 128

/*******************************************************************************
 * Types
 ******************************************************************************/

/**
 * @brief Device state machine.
 */
typedef enum {
  STATE_IDLE = 0,           /**< Waiting for connect command */
  STATE_CONNECTED,          /**< Connected, waiting for start command */
  STATE_STREAMING,          /**< Streaming raw EMG CSV to laptop (data collection) */
  STATE_PREDICTING,         /**< On-device inference + arm control */
  STATE_LAPTOP_PREDICT,     /**< Streaming CSV to laptop; laptop infers + sends gesture cmds back */
  STATE_CALIBRATING,        /**< Collecting rest data for calibration */
} device_state_t;

/**
 * @brief Commands from host.
 */
typedef enum {
  CMD_NONE = 0,
  CMD_CONNECT,
  CMD_START,                  /**< Start raw ADC streaming to laptop */
  CMD_START_PREDICT,          /**< Start on-device inference + arm control */
  CMD_START_LAPTOP_PREDICT,   /**< Start laptop-mediated inference (stream + receive cmds) */
  CMD_CALIBRATE,              /**< Run rest calibration (z-score + bicep threshold) */
  CMD_STOP,
  CMD_DISCONNECT,
} command_t;

/*******************************************************************************
 * Global State
 ******************************************************************************/

static volatile device_state_t g_device_state = STATE_IDLE;
static QueueHandle_t g_cmd_queue = NULL;

// Latest gesture command received from laptop during STATE_LAPTOP_PREDICT.
// Written by serial_input_task; read+cleared by run_laptop_predict_loop.
// gesture_t is a 32-bit int on LX7 — reads/writes are atomic; volatile is sufficient.
static volatile gesture_t g_laptop_gesture = GESTURE_NONE;

/*******************************************************************************
 * Forward Declarations
 ******************************************************************************/

static void send_ack_connect(void);
static gesture_t parse_laptop_gesture(const char *line);

/*******************************************************************************
 * Command Parsing
 ******************************************************************************/

/**
 * @brief Parse incoming command from JSON.
 *
 * Expected format: {"cmd": "connect"}
 *
 * @param line Input line from serial
 * @return Parsed command
 */
static command_t parse_command(const char *line) {
  /* Simple JSON parsing - look for "cmd" field */
  const char *cmd_start = strstr(line, "\"cmd\"");
  if (!cmd_start) {
    return CMD_NONE;
  }

  /* Find the value after "cmd": */
  const char *value_start = strchr(cmd_start, ':');
  if (!value_start) {
    return CMD_NONE;
  }

  /* Skip whitespace and opening quote */
  value_start++;
  while (*value_start == ' ' || *value_start == '"') {
    value_start++;
  }

  /* Match command strings — ordered longest-prefix-first to avoid false matches */
  if (strncmp(value_start, "connect", 7) == 0) {
    return CMD_CONNECT;
  } else if (strncmp(value_start, "start_laptop_predict", 20) == 0) {
    return CMD_START_LAPTOP_PREDICT;
  } else if (strncmp(value_start, "start_predict", 13) == 0) {
    return CMD_START_PREDICT;
  } else if (strncmp(value_start, "start", 5) == 0) {
    return CMD_START;
  } else if (strncmp(value_start, "calibrate", 9) == 0) {
    return CMD_CALIBRATE;
  } else if (strncmp(value_start, "stop", 4) == 0) {
    return CMD_STOP;
  } else if (strncmp(value_start, "disconnect", 10) == 0) {
    return CMD_DISCONNECT;
  }

  return CMD_NONE;
}

/*******************************************************************************
 * Laptop Gesture Parser
 ******************************************************************************/

/**
 * @brief Parse a gesture command sent by live_predict.py.
 *
 * Expected format: {"gesture":"fist"}
 * Returns GESTURE_NONE if the line is not a valid gesture command.
 */
static gesture_t parse_laptop_gesture(const char *line) {
  const char *g = strstr(line, "\"gesture\"");
  if (!g) return GESTURE_NONE;

  const char *v = strchr(g, ':');
  if (!v) return GESTURE_NONE;

  v++;
  while (*v == ' ' || *v == '"') v++;

  // Extract gesture name up to the closing quote
  char name[32] = {0};
  int ni = 0;
  while (*v && *v != '"' && ni < (int)(sizeof(name) - 1)) {
    name[ni++] = *v++;
  }

  // Delegate to the inference module's name→enum mapping
  int result = inference_get_gesture_by_name(name);
  return (gesture_t)result;
}

/*******************************************************************************
 * Serial Input Task
 ******************************************************************************/

/**
 * @brief FreeRTOS task to read serial input and parse commands.
 */
static void serial_input_task(void *pvParameters) {
  char line_buffer[CMD_BUFFER_SIZE];
  int line_idx = 0;

  while (1) {
    int c = getchar();

    if (c == EOF || c == 0xFF) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    if (c == '\n' || c == '\r') {
      if (line_idx > 0) {
        line_buffer[line_idx] = '\0';
        command_t cmd = parse_command(line_buffer);

        // When laptop-predict is active, try to parse gesture commands first.
        // These are separate from FSM commands and must not block the FSM parser.
        if (g_device_state == STATE_LAPTOP_PREDICT) {
          gesture_t g = parse_laptop_gesture(line_buffer);
          if (g != GESTURE_NONE) {
            g_laptop_gesture = g;
          }
        }

        if (cmd != CMD_NONE) {
          if (cmd == CMD_CONNECT) {
            g_device_state = STATE_CONNECTED;
            send_ack_connect();
            printf("[STATE] ANY -> CONNECTED (reconnect)\n");
          } else {
            switch (g_device_state) {
            case STATE_IDLE:
              break;

            case STATE_CONNECTED:
              if (cmd == CMD_START) {
                g_device_state = STATE_STREAMING;
                printf("[STATE] CONNECTED -> STREAMING\n");
                xQueueSend(g_cmd_queue, &cmd, 0);
              } else if (cmd == CMD_START_PREDICT) {
                g_device_state = STATE_PREDICTING;
                printf("[STATE] CONNECTED -> PREDICTING\n");
                xQueueSend(g_cmd_queue, &cmd, 0);
              } else if (cmd == CMD_START_LAPTOP_PREDICT) {
                g_device_state = STATE_LAPTOP_PREDICT;
                g_laptop_gesture = GESTURE_NONE;
                printf("[STATE] CONNECTED -> LAPTOP_PREDICT\n");
                xQueueSend(g_cmd_queue, &cmd, 0);
              } else if (cmd == CMD_CALIBRATE) {
                g_device_state = STATE_CALIBRATING;
                printf("[STATE] CONNECTED -> CALIBRATING\n");
                xQueueSend(g_cmd_queue, &cmd, 0);
              } else if (cmd == CMD_DISCONNECT) {
                g_device_state = STATE_IDLE;
                printf("[STATE] CONNECTED -> IDLE\n");
              }
              break;

            case STATE_STREAMING:
            case STATE_PREDICTING:
            case STATE_LAPTOP_PREDICT:
            case STATE_CALIBRATING:
              if (cmd == CMD_STOP) {
                g_device_state = STATE_CONNECTED;
                printf("[STATE] ACTIVE -> CONNECTED\n");
              } else if (cmd == CMD_DISCONNECT) {
                g_device_state = STATE_IDLE;
                printf("[STATE] ACTIVE -> IDLE\n");
              }
              break;
            }
          }
        }
        line_idx = 0;
      }
    } else if (line_idx < CMD_BUFFER_SIZE - 1) {
      line_buffer[line_idx++] = (char)c;
    } else {
      line_idx = 0;
    }
  }
}

/*******************************************************************************
 * State Machine
 ******************************************************************************/

static void send_ack_connect(void) {
  printf(
      "{\"status\":\"ack_connect\",\"device\":\"ESP32-EMG\",\"channels\":%d}\n",
      EMG_NUM_CHANNELS);
  fflush(stdout);
}

/**
 * @brief Stream raw EMG data (Training Mode).
 */
static void stream_emg_data(void) {
  emg_sample_t sample;

  while (g_device_state == STATE_STREAMING) {
    emg_sensor_read(&sample);  // blocks ~1 ms (queue-paced by DMA task)
    printf("%u,%u,%u,%u\n", sample.channels[0], sample.channels[1],
           sample.channels[2], sample.channels[3]);
  }
}

/*******************************************************************************
 * Multi-Model Voting Post-Processing
 *
 * Shared EMA smoothing, majority vote, and debounce applied to the averaged
 * probability output from all enabled models (single LDA, ensemble, MLP).
 ******************************************************************************/

/* Retuned 2026-07-13. Previously: EMA + a 5-wide majority vote + a 3-hit debounce.
 * The EMA already smooths, so the vote and debounce were double-smoothing — they
 * added latency without buying stability, and the output still flickered at 1.8x
 * the true gesture-change rate.
 *
 * Replaced with EMA + HYSTERESIS: it takes a lot of confidence to ADOPT a new
 * gesture (ENTER) but very little to KEEP the current one (HOLD). That asymmetry
 * is what actually suppresses flicker.
 *
 * Measured on 4 held-out sessions (96 true gesture changes), steady-state accuracy
 * vs. output switches:
 *   old  EMA+vote5+debounce3 : 85.48%, 175 switches (1.82x)
 *   new  EMA+hysteresis      : 87.25%, 115 switches (1.20x)
 * Better accuracy AND ~35% less flicker. */
#define VOTE_EMA_ALPHA      0.70f
#define VOTE_ENTER_THRESHOLD 0.70f  /* confidence required to SWITCH to a new gesture */
#define VOTE_HOLD_THRESHOLD  0.25f  /* below this we hold the last output, unconditionally */

static float vote_smoothed[MODEL_NUM_CLASSES];
static int   vote_current_output = -1;

static void vote_init(void) {
  for (int c = 0; c < MODEL_NUM_CLASSES; c++)
    vote_smoothed[c] = 1.0f / MODEL_NUM_CLASSES;
  vote_current_output = -1;
}

/**
 * @brief Apply EMA + hysteresis to averaged probabilities.
 *
 * @param avg_proba  Averaged probability vector from all models.
 * @param confidence Output: smoothed confidence of the final winner.
 * @return Final gesture class index (-1 before the first confident lock).
 */
static int vote_postprocess(const float *avg_proba, float *confidence) {
  float max_smooth = 0.0f;
  int winner = 0;
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
    vote_smoothed[c] = VOTE_EMA_ALPHA * vote_smoothed[c] +
                       (1.0f - VOTE_EMA_ALPHA) * avg_proba[c];
    if (vote_smoothed[c] > max_smooth) {
      max_smooth = vote_smoothed[c];
      winner = c;
    }
  }
  *confidence = max_smooth;

  /* Too unsure to say anything — keep doing whatever we were doing. */
  if (max_smooth < VOTE_HOLD_THRESHOLD)
    return vote_current_output;

  /* Already holding this gesture: no threshold to clear, just stay. */
  if (winner == vote_current_output)
    return vote_current_output;

  /* Changing gesture is the expensive move — demand real confidence for it. */
  if (vote_current_output < 0 || max_smooth >= VOTE_ENTER_THRESHOLD)
    vote_current_output = winner;

  return vote_current_output;
}

/**
 * @brief Run on-device inference (Prediction Mode).
 *
 * Uses multi-model voting: all enabled models (single LDA, ensemble, MLP)
 * produce raw probability vectors which are averaged, then passed through
 * shared EMA smoothing, majority vote, and debounce.
 */
static void run_inference_loop(void) {
  emg_sample_t sample;
  int stride_counter = 0;

  inference_init();

#if ENABLE_HAND
  int last_gesture = -1;
  vote_init();

  int n_models = 1;  /* single LDA always enabled */
#if MODEL_USE_ENSEMBLE
  inference_ensemble_init();
  n_models++;
#endif
#if MODEL_USE_MLP
  inference_mlp_init();
  n_models++;
#endif

  printf("{\"status\":\"info\",\"msg\":\"Multi-model inference (%d models)\"}\n",
         n_models);
#endif /* ENABLE_HAND */

  while (g_device_state == STATE_PREDICTING) {
    emg_sensor_read(&sample);

    if (inference_add_sample(sample.channels)) {
      stride_counter++;

      if (stride_counter >= INFERENCE_HOP_SIZE) {
        stride_counter = 0;

#if ENABLE_HAND
        /* 1. Extract features once */
        float features[MODEL_NUM_FEATURES];
        inference_extract_features(features);

        /* 2. Collect raw probabilities from each model */
        float avg_proba[MODEL_NUM_CLASSES];
        memset(avg_proba, 0, sizeof(avg_proba));

        /* Model A: Single LDA (always active) */
        float proba_lda[MODEL_NUM_CLASSES];
        inference_predict_raw(features, proba_lda);
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
          avg_proba[c] += proba_lda[c];

#if MODEL_USE_ENSEMBLE
        /* Model B: 3-specialist + meta-LDA ensemble */
        float proba_ens[MODEL_NUM_CLASSES];
        inference_ensemble_predict_raw(features, proba_ens);
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
          avg_proba[c] += proba_ens[c];
#endif

#if MODEL_USE_MLP
        /* Model C: int8 MLP — returns class index + confidence.
         * Spread confidence as soft one-hot for averaging. */
        float mlp_conf = 0.0f;
        int mlp_class = inference_mlp_predict(features, MODEL_NUM_FEATURES, &mlp_conf);
        float remainder = (1.0f - mlp_conf) / (MODEL_NUM_CLASSES - 1);
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
          avg_proba[c] += (c == mlp_class) ? mlp_conf : remainder;
#endif

        /* 3. Average across models */
        float inv_n = 1.0f / n_models;
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
          avg_proba[c] *= inv_n;

        /* 4. Shared post-processing */
        float confidence = 0.0f;
        int gesture_idx = vote_postprocess(avg_proba, &confidence);

        if (gesture_idx >= 0) {
          int gesture_enum = inference_get_gesture_enum(gesture_idx);
          gestures_execute((gesture_t)gesture_enum);

          if (gesture_idx != last_gesture) {
            printf("{\"gesture\":\"%s\",\"conf\":%.2f}\n",
                   inference_get_class_name(gesture_idx), confidence);
            last_gesture = gesture_idx;
          }
        }
#endif /* ENABLE_HAND */

#if ENABLE_BICEP
#if BICEP_PROPORTIONAL
        bicep_apply_proportional();
#else
        bicep_apply(bicep_detect());
#endif
#endif
      }
    }
  }
}

/**
 * @brief Laptop-mediated prediction loop (STATE_LAPTOP_PREDICT).
 *
 * Streams raw ADC CSV to laptop at 1kHz (same format as STATE_STREAMING).
 * The laptop runs live_predict.py, which classifies each window and sends
 * {"gesture":"<name>"} back over UART. serial_input_task() intercepts those
 * lines and writes to g_laptop_gesture; this loop executes whatever is set.
 */
static void run_laptop_predict_loop(void) {
  emg_sample_t sample;
  gesture_t last_gesture = GESTURE_NONE;

  printf("{\"status\":\"info\",\"msg\":\"Laptop-predict mode started\"}\n");

  while (g_device_state == STATE_LAPTOP_PREDICT) {
    emg_sensor_read(&sample);
    // Stream raw CSV to laptop (live_predict.py reads this)
    printf("%u,%u,%u,%u\n", sample.channels[0], sample.channels[1],
           sample.channels[2], sample.channels[3]);

    // Execute any gesture command that arrived from the laptop
    gesture_t g = g_laptop_gesture;
    if (g != GESTURE_NONE) {
      g_laptop_gesture = GESTURE_NONE;  // clear before executing
      gestures_execute(g);
      if (g != last_gesture) {
        // Echo the executed gesture back for laptop-side logging
        printf("{\"executed\":\"%s\"}\n", gestures_get_name(g));
        last_gesture = g;
      }
    }
  }
}

/**
 * @brief Fully autonomous inference loop (MAIN_MODE == EMG_STANDALONE).
 *
 * No laptop required. Runs forever until power-off.
 * Assumes inference_init() and sensor init have already been called by app_main().
 */
static void run_standalone_loop(void) {
  emg_sample_t sample;
  int stride_counter = 0;

  inference_init();

#if ENABLE_HAND
  int last_gesture = -1;
  vote_init();

  int n_models = 1;
#if MODEL_USE_ENSEMBLE
  inference_ensemble_init();
  n_models++;
#endif
#if MODEL_USE_MLP
  inference_mlp_init();
  n_models++;
#endif

  printf("[STANDALONE] Running autonomous EMG control (%d models).\n", n_models);
#endif /* ENABLE_HAND */

  while (1) {
    emg_sensor_read(&sample);

    if (inference_add_sample(sample.channels)) {
      stride_counter++;
      if (stride_counter >= INFERENCE_HOP_SIZE) {
        stride_counter = 0;

#if ENABLE_HAND
        float features[MODEL_NUM_FEATURES];
        inference_extract_features(features);

        float avg_proba[MODEL_NUM_CLASSES];
        memset(avg_proba, 0, sizeof(avg_proba));

        float proba_lda[MODEL_NUM_CLASSES];
        inference_predict_raw(features, proba_lda);
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
          avg_proba[c] += proba_lda[c];

#if MODEL_USE_ENSEMBLE
        float proba_ens[MODEL_NUM_CLASSES];
        inference_ensemble_predict_raw(features, proba_ens);
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
          avg_proba[c] += proba_ens[c];
#endif

#if MODEL_USE_MLP
        float mlp_conf = 0.0f;
        int mlp_class = inference_mlp_predict(features, MODEL_NUM_FEATURES,
                                              &mlp_conf);
        float remainder = (1.0f - mlp_conf) / (MODEL_NUM_CLASSES - 1);
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
          avg_proba[c] += (c == mlp_class) ? mlp_conf : remainder;
#endif

        float inv_n = 1.0f / n_models;
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
          avg_proba[c] *= inv_n;
        float confidence = 0.0f;
        int gesture_idx = vote_postprocess(avg_proba, &confidence);

        if (gesture_idx >= 0) {
          int gesture_enum = inference_get_gesture_enum(gesture_idx);
          gestures_execute((gesture_t)gesture_enum);

          if (gesture_idx != last_gesture) {
            printf("{\"gesture\":\"%s\",\"conf\":%.2f,\"t_ms\":%lu}\n",
                   inference_get_class_name(gesture_idx), confidence,
                   (unsigned long)sample.timestamp_ms);
            for (int c = 0; c < MODEL_NUM_CLASSES; c++)
              printf("%s:%.2f\n", inference_get_class_name(c), avg_proba[c]);
            last_gesture = gesture_idx;
          }
        }
#endif /* ENABLE_HAND */

#if ENABLE_BICEP
#if BICEP_PROPORTIONAL
        bicep_apply_proportional();
#else
        bicep_apply(bicep_detect());
#endif
#endif
      }
    }
  }
}

/**
 * @brief Run rest calibration (STATE_CALIBRATING).
 *
 * Collects 3 seconds of rest EMG data, extracts features from each window,
 * then updates:
 *   - NVS z-score calibration (Change D) via calibration_update()
 *   - Bicep detection threshold via bicep_calibrate_from_buffer()
 *
 * The user must keep their arm relaxed during this period.
 * Send {"cmd": "calibrate"} from the host to trigger.
 */
static void run_calibration(void) {
#if LIVE_EMG
  #define CALIB_DURATION_SAMPLES 5000  /* 3 seconds at 1 kHz */
#else
  #define CALIB_DURATION_SAMPLES 2000
#endif
  #define CALIB_MAX_WINDOWS \
((CALIB_DURATION_SAMPLES - INFERENCE_WINDOW_SIZE) / INFERENCE_HOP_SIZE + 1)

  /* Discarded after each on-screen prompt: gives the user time to react and
   * lets the muscle's rest↔flex transition pass so it never taints the RMS. */
  #define BICEP_CALIB_SETTLE_SAMPLES 1500  /* 1.5 s */
  #define BICEP_CALIB_MAX_SAMPLES    2000  /* 2 s of sustained max flex */

  printf("{\"status\":\"calibrating\",\"duration_ms\":%d}\n", CALIB_DURATION_SAMPLES);
  fflush(stdout);

  inference_init();      /* reset buffer + filter state */
  calibration_reset();   /* collect RAW features — don't z-score with stale stats */

  float *feat_matrix = (float *)malloc(
      (size_t)CALIB_MAX_WINDOWS * MODEL_NUM_FEATURES * sizeof(float));
  if (!feat_matrix) {
    printf("{\"status\":\"error\",\"msg\":\"Calibration malloc failed\"}\n");
    g_device_state = STATE_CONNECTED;
    return;
  }

  emg_sample_t sample;
  int window_count   = 0;
  int stride_counter = 0;

#if ENABLE_BICEP && BICEP_PROPORTIONAL
  float bicep_rest_sum = 0.0f; int bicep_rest_n = 0;
  float bicep_max_sum  = 0.0f; int bicep_max_n  = 0;

  /* ---- Prompt: REST phase, then settle out the transition ---- */
  printf("\n================ CALIBRATION ================\n");
  printf(">>> REST calibration starting.\n");
  printf(">>> RELAX your arm completely and keep it still.\n");
  printf("============================================\n");
  printf("{\"status\":\"calib_phase\",\"phase\":\"rest\"}\n");
  fflush(stdout);
  for (int s = 0; s < BICEP_CALIB_SETTLE_SAMPLES; s++) {
    emg_sensor_read(&sample);
    inference_add_sample(sample.channels);  /* warm up filter/buffer, discard */
  }
#endif

  /* ---- REST collection: hand z-score features + bicep rest RMS ---- */
  for (int s = 0; s < CALIB_DURATION_SAMPLES; s++) {
    emg_sensor_read(&sample);
    if (inference_add_sample(sample.channels)) {
      stride_counter++;
      if (stride_counter >= INFERENCE_HOP_SIZE) {
        stride_counter = 0;
        if (window_count < CALIB_MAX_WINDOWS) {
          inference_extract_features(
              feat_matrix + window_count * MODEL_NUM_FEATURES);
          window_count++;
        }
#if ENABLE_BICEP && BICEP_PROPORTIONAL
        bicep_rest_sum += bicep_current_rms();
        bicep_rest_n++;
#endif
      }
    }
  }

#if ENABLE_BICEP && BICEP_PROPORTIONAL
  /* ---- Prompt: MAX phase, settle out the ramp-up, then collect ---- */
  printf("\n============================================\n");
  printf(">>> MAX-FLEX calibration starting.\n");
  printf(">>> FLEX your bicep as HARD as you can and HOLD it.\n");
  printf("============================================\n");
  printf("{\"status\":\"calib_phase\",\"phase\":\"max\"}\n");
  fflush(stdout);
  for (int s = 0; s < BICEP_CALIB_SETTLE_SAMPLES; s++) {
    emg_sensor_read(&sample);
    inference_add_sample(sample.channels);  /* let the ramp to full flex pass */
  }

  int bicep_stride = 0;
  for (int s = 0; s < BICEP_CALIB_MAX_SAMPLES; s++) {
    emg_sensor_read(&sample);
    if (inference_add_sample(sample.channels)) {
      if (++bicep_stride >= INFERENCE_HOP_SIZE) {
        bicep_stride = 0;
        bicep_max_sum += bicep_current_rms();
        bicep_max_n++;
      }
    }
  }
  printf(">>> You can relax now.\n\n");
  fflush(stdout);
#endif

  if (window_count >= 10) {
    calibration_update(feat_matrix, window_count, MODEL_NUM_FEATURES);
#if ENABLE_BICEP && BICEP_PROPORTIONAL
    if (bicep_rest_n > 0 && bicep_max_n > 0) {
      float rest_rms = bicep_rest_sum / bicep_rest_n;
      float max_rms  = bicep_max_sum  / bicep_max_n;
      if (max_rms > rest_rms * 1.2f) {
        bicep_set_proportional(rest_rms, max_rms);
        bicep_save_proportional(rest_rms, max_rms);
      } else {
        printf("{\"status\":\"error\",\"msg\":\"Bicep max not distinct from rest "
               "(rest=%.2f max=%.2f) — flex harder and recalibrate\"}\n",
               rest_rms, max_rms);
      }
    }
#else
    bicep_calibrate_from_buffer(INFERENCE_WINDOW_SIZE);
#endif
    printf("{\"status\":\"calibrated\",\"windows\":%d}\n", window_count);
  } else {
    printf("{\"status\":\"error\",\"msg\":\"Not enough calibration data\"}\n");
  }

  free(feat_matrix);
  g_device_state = STATE_CONNECTED;

  #undef CALIB_DURATION_SAMPLES
  #undef CALIB_MAX_WINDOWS
  #undef BICEP_CALIB_SETTLE_SAMPLES
  #undef BICEP_CALIB_MAX_SAMPLES
}

#if MAIN_MODE == PARITY_DUMP
/**
 * @brief Train/serve parity harness (MAIN_MODE == PARITY_DUMP).
 *
 * Walks replay_samples[] directly — no esp_timer, no queue, no real-time pacing —
 * so the run is fully deterministic and blocking printf cannot drop samples.
 * Feeds the exact same input Python gets from the source HDF5, then dumps every
 * quantity needed to diff the two implementations feature-by-feature.
 *
 * Output is line-oriented and machine-parsable; see tools/parity_compare.py.
 */
static void run_parity_dump(void) {
  const int CALIB_SAMPLES = 2000; /* mirrors run_calibration() in replay mode */
  const int CALIB_MAX_WIN =
      (CALIB_SAMPLES - INFERENCE_WINDOW_SIZE) / INFERENCE_HOP_SIZE + 1;

  /* ---- Phase 1: calibrate off the replay's leading rest period ---- */
  inference_init();
  calibration_reset(); /* collect RAW features */

  float *feat_matrix =
      (float *)malloc((size_t)CALIB_MAX_WIN * MODEL_NUM_FEATURES * sizeof(float));
  if (!feat_matrix) {
    printf("#error malloc failed\n");
    return;
  }

  int win = 0, stride = 0;
  for (int s = 0; s < CALIB_SAMPLES && s < REPLAY_NUM_SAMPLES; s++) {
    uint16_t ch[NUM_CHANNELS];
    for (int c = 0; c < NUM_CHANNELS; c++)
      ch[c] = replay_samples[s][c];
    if (inference_add_sample(ch)) {
      if (++stride >= INFERENCE_HOP_SIZE) {
        stride = 0;
        if (win < CALIB_MAX_WIN) {
          inference_extract_features(feat_matrix + win * MODEL_NUM_FEATURES);
          win++;
        }
      }
    }
  }
  calibration_update(feat_matrix, win, MODEL_NUM_FEATURES);
  free(feat_matrix);

  float cmean[MODEL_NUM_FEATURES], cstd[MODEL_NUM_FEATURES];
  int nf = calibration_get_stats(cmean, cstd);

#if MODEL_USE_ENSEMBLE
  inference_ensemble_init();
#endif
#if MODEL_USE_MLP
  inference_mlp_init();
#endif

  /* The dump is deterministic, so just repeat it forever: the capture tool can
     attach at any time and still get one complete, self-contained cycle without
     anyone having to hit RESET with the right timing. */
  while (1) {
    printf("#PARITY_BEGIN\n");
    printf("#meta n_feat=%d n_class=%d win=%d hop=%d replay_n=%d hops=%d\n",
           MODEL_NUM_FEATURES, MODEL_NUM_CLASSES, INFERENCE_WINDOW_SIZE,
           INFERENCE_HOP_SIZE, REPLAY_NUM_SAMPLES, PARITY_DUMP_HOPS);
    printf("#calib_windows %d\n", win);
    printf("#calib_valid %d\n", nf);

    /* Calibration vectors: let Python reproduce calibration_apply() and verify
       the sigma_train override + A2 mean delta landed correctly. */
    printf("CMEAN");
    for (int i = 0; i < nf; i++) printf(",%.9g", cmean[i]);
    printf("\nCSTD");
    for (int i = 0; i < nf; i++) printf(",%.9g", cstd[i]);
    printf("\n");

    /* ---- Phase 2: replay from sample 0 and dump every hop ---- */
    inference_init(); /* reset filter state + window buffer */

    int hop = 0;
    stride = 0;
    for (int s = 0; s < REPLAY_NUM_SAMPLES && hop < PARITY_DUMP_HOPS; s++) {
      uint16_t ch[NUM_CHANNELS];
      for (int c = 0; c < NUM_CHANNELS; c++)
        ch[c] = replay_samples[s][c];

      if (!inference_add_sample(ch))
        continue;
      if (++stride < INFERENCE_HOP_SIZE)
        continue;
      stride = 0;

      float feat[MODEL_NUM_FEATURES];
      inference_extract_features(feat); /* calibrated */

      float p_lda[MODEL_NUM_CLASSES];
      inference_predict_raw(feat, p_lda);

      /* H,<hop>,<end_sample_idx> — end_sample_idx is the index of the LAST
         sample in this window, so Python's window is raw[idx-149 .. idx]. */
      printf("H,%d,%d", hop, s);
      for (int i = 0; i < MODEL_NUM_FEATURES; i++) printf(",%.9g", feat[i]);
      for (int c = 0; c < MODEL_NUM_CLASSES; c++)  printf(",%.9g", p_lda[c]);

#if MODEL_USE_ENSEMBLE
      float p_ens[MODEL_NUM_CLASSES];
      inference_ensemble_predict_raw(feat, p_ens);
      for (int c = 0; c < MODEL_NUM_CLASSES; c++) printf(",%.9g", p_ens[c]);
#else
      for (int c = 0; c < MODEL_NUM_CLASSES; c++) printf(",nan");
#endif

#if MODEL_USE_MLP
      float mlp_conf = 0.0f;
      int mlp_class = inference_mlp_predict(feat, MODEL_NUM_FEATURES, &mlp_conf);
      printf(",%d,%.9g", mlp_class, mlp_conf);
#else
      printf(",-1,nan");
#endif
      printf("\n");
      hop++;
    }

    printf("#PARITY_END %d\n", hop);
    fflush(stdout);
    vTaskDelay(pdMS_TO_TICKS(3000));
  }
}
#endif /* MAIN_MODE == PARITY_DUMP */

static void state_machine_loop(void) {
  command_t cmd;
  const TickType_t poll_interval = pdMS_TO_TICKS(50);

  while (1) {
    if (g_device_state == STATE_STREAMING) {
      stream_emg_data();
    } else if (g_device_state == STATE_PREDICTING) {
      run_inference_loop();
    } else if (g_device_state == STATE_LAPTOP_PREDICT) {
      run_laptop_predict_loop();
    } else if (g_device_state == STATE_CALIBRATING) {
      run_calibration();
    }

    xQueueReceive(g_cmd_queue, &cmd, poll_interval);
  }
}

void appConnector() {
  g_cmd_queue = xQueueCreate(10, sizeof(command_t));
  if (g_cmd_queue == NULL) {
    printf("[ERROR] Failed to create command queue!\n");
    return;
  }

  xTaskCreate(serial_input_task, "serial_input", 4096, NULL, 5, NULL);

  printf("[PROTOCOL] Waiting for host to connect...\n");
  printf("[PROTOCOL] Send: {\"cmd\": \"connect\"}\n");
  printf("[PROTOCOL] Send: {\"cmd\": \"start_predict\"} for on-device "
         "inference\n");
  printf("[PROTOCOL] Send: {\"cmd\": \"calibrate\"} for rest "
         "calibration (3s)\n\n");

  state_machine_loop();
}

/*******************************************************************************
 * Application Entry Point
 ******************************************************************************/

void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(3000));  /* Wait for monitor to connect after upload */
  printf("\n");
  printf("========================================\n");
  printf("  Bucky Arm - EMG Robotic Hand\n");
  printf("  Firmware v2.1.0 (Inference Enabled)\n");
  printf("========================================\n\n");

  printf("[INIT] Initializing hand (servos)...\n");
  hand_init();

  printf("[INIT] Initializing EMG sensor...\n");
  emg_sensor_init();

  printf("[INIT] Initializing Inference Engine...\n");
  inference_init();

  printf("[INIT] Loading NVS calibration...\n");
  calibration_init();  // Change D: no-op on first boot; loads if previously saved

  // Bicep: load persisted calibration from NVS (if previously calibrated)
  {
#if ENABLE_BICEP && BICEP_PROPORTIONAL
    float bicep_rest = 0.0f, bicep_max = 0.0f;
    if (bicep_load_proportional(&bicep_rest, &bicep_max)) {
      printf("[INIT] Bicep proportional calib loaded: rest=%.2f max=%.2f\n",
             bicep_rest, bicep_max);
    } else {
      printf("[INIT] No bicep proportional calibration — run 'calibrate' "
             "command (rest + max-flex)\n");
    }
#else
    float bicep_thresh = 0.0f;
    if (bicep_load_threshold(&bicep_thresh)) {
      printf("[INIT] Bicep threshold loaded: %.1f\n", bicep_thresh);
    } else {
      printf("[INIT] No bicep calibration — run 'calibrate' command\n");
    }
#endif
  }

  printf("[INIT] Using REAL EMG sensors\n");
  printf("[INIT] Done!\n\n");

  switch (MAIN_MODE) {
#if MAIN_MODE == PARITY_DUMP
  case PARITY_DUMP:
    run_parity_dump();  // never returns
    break;
#endif

  case EMG_STANDALONE:
    // Fully autonomous: no laptop needed after this point.
    // Boots directly into the inference + arm control loop.
    run_standalone_loop();  // never returns
    break;

  case SERVO_CALIBRATOR_ANGLE:
    while (1) {
      int angle;
      printf("Enter servo angle (0-180): ");
      fflush(stdout);

      // Read a line manually, yielding while waiting for UART input
      char buf[16];
      int idx = 0;
      while (idx < (int)sizeof(buf) - 1) {
        int ch = getchar();
        if (ch == EOF) {
          vTaskDelay(pdMS_TO_TICKS(10));
          continue;
        }
        if (ch == '\n' || ch == '\r')
          break;
        buf[idx++] = (char)ch;
      }
      buf[idx] = '\0';

      if (idx == 0)
        continue;

      if (sscanf(buf, "%d", &angle) == 1) {
        hand_set_finger_angle(JOINT_BICEP, angle);
          vTaskDelay(pdMS_TO_TICKS(1000));
      } else {
        printf("Invalid input.\n");
      }
    }
    break;

  case SERVO_CALIBRATOR_DUTY:
    while (1) {
      int duty;
      printf("Enter servo duty: ");
      fflush(stdout);

      // Read a line manually, yielding while waiting for UART input
      char buf[16];
      int idx = 0;
      while (idx < (int)sizeof(buf) - 1) {
        int ch = getchar();
        if (ch == EOF) {
          vTaskDelay(pdMS_TO_TICKS(10));
          continue;
        }
        if (ch == '\n' || ch == '\r')
          break;
        buf[idx++] = (char)ch;
      }
      buf[idx] = '\0';

      if (idx == 0)
        continue;

      if (sscanf(buf, "%d", &duty) == 1) {
        hand_set_finger_duty(JOINT_THUMB, duty);
          vTaskDelay(pdMS_TO_TICKS(1000));
      } else {
        printf("Invalid input.\n");
      }
    }
    break;

  case GESTURE_TESTER:
    while (1) {
      fflush(stdout);

      int ch = getchar();

      if (ch == EOF) {
        vTaskDelay(pdMS_TO_TICKS(10));
        continue;
      }

      if (ch == '\n' || ch == '\r') {
        continue;
      }

      gesture_t gesture = GESTURE_NONE;

      switch (ch) {
        case 'r': gesture = GESTURE_REST; break;
        case 'f': gesture = GESTURE_FIST; break;
        case 'o': gesture = GESTURE_OPEN; break;
        case 'h': gesture = GESTURE_HOOK_EM; break;
        case 't': gesture = GESTURE_THUMBS_UP; break;
        default:
          printf("Invalid gesture: %c\n", ch);
          continue;
      }

      printf("Executing gesture: %s\n", gestures_get_name(gesture));
      gestures_execute(gesture);

      vTaskDelay(pdMS_TO_TICKS(500));
    }

    break;

  case CONTINUOUS_TEST: {
    // Continuous rotation servo: the PWM "angle" sets speed and direction.
    //   ~90 deg (neutral) = stopped
    //    0 deg (SERVO_DUTY_MIN, ~1ms) = full speed one direction
    //  180 deg (SERVO_DUTY_MAX, ~2ms) = full speed other direction
    // Flip CONT_SERVO_CW_ANGLE to 180.0f if the servo spins the wrong way.
    #define CONT_SERVO_FINGER    JOINT_THUMB
    #define CONT_SERVO_CW_ANGLE  0.0f

    printf("[CONTINUOUS_TEST] Driving servo clockwise at constant speed...\n");
    printf("[CONTINUOUS_TEST] Finger: %d, Angle: %.0f\n",
           CONT_SERVO_FINGER, CONT_SERVO_CW_ANGLE);

    hand_set_finger_angle(CONT_SERVO_FINGER, CONT_SERVO_CW_ANGLE);

    while (1) {
      vTaskDelay(pdMS_TO_TICKS(1000));
    }
    break;
  }

  case EMG_MAIN:
    appConnector();
    break;

  case REAL_MAIN:
    run_calibration();
    run_standalone_loop();
    break;
    
  default:
    printf("[ERROR] Unknown MAIN_MODE\n");
    break;
  }
}