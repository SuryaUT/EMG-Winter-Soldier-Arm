/**
 * @file emg_sensor.c
 * @brief EMG sensor driver — DMA-backed continuous ADC acquisition.
 *
 * Change A: adc_continuous (DMA) replaces adc_oneshot polling.
 * A background FreeRTOS task on Core 0 reads DMA frames, assembles
 * complete 4-channel sample sets, and pushes them to a FreeRTOS queue.
 * emg_sensor_read() blocks on that queue; at 1 kHz per channel this
 * returns within ~1 ms, providing exact timing without vTaskDelay(1).
 */

#include "emg_sensor.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_adc/adc_continuous.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_err.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// --- ADC DMA constants ---
// Total sample rate: 4 channels × 1 kHz = 4 kHz
#define ADC_TOTAL_SAMPLE_RATE_HZ  (EMG_NUM_CHANNELS * EMG_SAMPLE_RATE_HZ)
// DMA frame: 64 conversions × 4 bytes = 256 bytes, arrives ~every 16 ms
#define ADC_CONV_FRAME_SIZE       (64u * sizeof(adc_digi_output_data_t))
// Internal DMA pool: 2× conv_frame_size
#define ADC_POOL_SIZE             (2u * ADC_CONV_FRAME_SIZE)
// Sample queue: buffers up to ~32 ms of assembled 4-channel sets
#define SAMPLE_QUEUE_DEPTH        32

// --- Static handles ---
static adc_continuous_handle_t s_adc_handle   = NULL;
static adc_cali_handle_t       s_cali_handle  = NULL;
static QueueHandle_t           s_sample_queue = NULL;

// Channel mapping (ADC1, GPIO 2/3/9/10)
static const adc_channel_t s_channels[EMG_NUM_CHANNELS] = {
    ADC_CHANNEL_1,  // GPIO 2  — FCR/Belly (ch0)
    ADC_CHANNEL_2,  // GPIO 3  — Extensors (ch1)
    ADC_CHANNEL_8,  // GPIO 9  — FCU/Outer Flexors (ch2)
    ADC_CHANNEL_9,  // GPIO 10 — Bicep (ch3)
};

/*******************************************************************************
 * ADC Sampling Task (Core 0)
 ******************************************************************************/

/**
 * @brief DMA sampling task.
 *
 * Reads adc_continuous DMA frames, assembles complete 4-channel sample sets
 * (one value per channel), applies curve-fitting calibration, and pushes
 * emg_sample_t structs to s_sample_queue. Runs continuously on Core 0.
 */
static void adc_sampling_task(void *arg) {
    uint8_t *buf = (uint8_t *)malloc(ADC_CONV_FRAME_SIZE);
    if (!buf) {
        printf("[EMG] ERROR: DMA read buffer alloc failed\n");
        vTaskDelete(NULL);
        return;
    }

    // Per-channel raw accumulator; holds the latest sample for each channel
    int  raw_set[EMG_NUM_CHANNELS];
    bool got[EMG_NUM_CHANNELS];
    memset(raw_set, 0, sizeof(raw_set));
    memset(got,     0, sizeof(got));

    while (1) {
        uint32_t out_len = 0;
        esp_err_t err = adc_continuous_read(
            s_adc_handle, buf, (uint32_t)ADC_CONV_FRAME_SIZE,
            &out_len, pdMS_TO_TICKS(100)
        );
        if (err != ESP_OK || out_len == 0) {
            continue;
        }

        uint32_t n = out_len / sizeof(adc_digi_output_data_t);
        adc_digi_output_data_t *p = (adc_digi_output_data_t *)buf;

        for (uint32_t i = 0; i < n; i++) {
            int ch  = (int)p[i].type2.channel;
            int raw = (int)p[i].type2.data;

            // Map ADC channel index to sensor channel
            for (int c = 0; c < EMG_NUM_CHANNELS; c++) {
                if ((int)s_channels[c] == ch) {
                    raw_set[c] = raw;
                    got[c]     = true;
                    break;
                }
            }

            // Emit a complete sample set once all channels have been updated
            bool all = true;
            for (int c = 0; c < EMG_NUM_CHANNELS; c++) {
                if (!got[c]) { all = false; break; }
            }
            if (!all) continue;

            emg_sample_t s;
            s.timestamp_ms = emg_sensor_get_timestamp_ms();
            for (int c = 0; c < EMG_NUM_CHANNELS; c++) {
                int mv = 0;
                adc_cali_raw_to_voltage(s_cali_handle, raw_set[c], &mv);
                s.channels[c] = (uint16_t)mv;
            }

            // Non-blocking send: drop if queue is full (prefer fresh data)
            xQueueSend(s_sample_queue, &s, 0);

            // Reset accumulator for the next complete set
            memset(got, 0, sizeof(got));
        }
    }
}

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

void emg_sensor_init(void) {
    // 1. Curve-fitting calibration (same scheme as before)
    adc_cali_curve_fitting_config_t cali_cfg = {
        .unit_id  = ADC_UNIT_1,
        .atten    = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    ESP_ERROR_CHECK(adc_cali_create_scheme_curve_fitting(&cali_cfg, &s_cali_handle));

    // 2. Create continuous ADC handle
    adc_continuous_handle_cfg_t adc_cfg = {
        .max_store_buf_size = ADC_POOL_SIZE,
        .conv_frame_size    = ADC_CONV_FRAME_SIZE,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_cfg, &s_adc_handle));

    // 3. Configure scan pattern (4 channels, 4 kHz total)
    adc_digi_pattern_config_t patterns[EMG_NUM_CHANNELS];
    for (int i = 0; i < EMG_NUM_CHANNELS; i++) {
        patterns[i].atten     = ADC_ATTEN_DB_12;
        patterns[i].channel   = (uint8_t)s_channels[i];
        patterns[i].unit      = ADC_UNIT_1;
        patterns[i].bit_width = ADC_BITWIDTH_12;
    }
    adc_continuous_config_t cont_cfg = {
        .pattern_num    = EMG_NUM_CHANNELS,
        .adc_pattern    = patterns,
        .sample_freq_hz = ADC_TOTAL_SAMPLE_RATE_HZ,
        .conv_mode      = ADC_CONV_SINGLE_UNIT_1,
        .format         = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
    };
    ESP_ERROR_CHECK(adc_continuous_config(s_adc_handle, &cont_cfg));

    // 4. Start DMA acquisition
    ESP_ERROR_CHECK(adc_continuous_start(s_adc_handle));

    // 5. Create sample queue (assembled complete sets land here)
    s_sample_queue = xQueueCreate(SAMPLE_QUEUE_DEPTH, sizeof(emg_sample_t));
    assert(s_sample_queue != NULL);

    // 6. Launch sampling task pinned to Core 0
    xTaskCreatePinnedToCore(adc_sampling_task, "adc_sample", 4096, NULL, 6, NULL, 0);
}

void emg_sensor_read(emg_sample_t *sample) {
    // Block until a complete 4-channel sample set arrives from the DMA task.
    // At 1 kHz per channel this typically returns within ~1 ms.
    xQueueReceive(s_sample_queue, sample, portMAX_DELAY);
}

uint32_t emg_sensor_get_timestamp_ms(void) {
    return (uint32_t)(esp_timer_get_time() / 1000);
}
