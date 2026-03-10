/**
 * @file servo_hal.c
 * @brief Hardware Abstraction Layer for servo PWM control.
 *
 * Implements low-level LEDC peripheral configuration and control.
 */

#include "servo_hal.h"
#include "driver/ledc.h"
#include "esp_err.h"

/*******************************************************************************
 * Private Data
 ******************************************************************************/

/** @brief GPIO pin for each finger servo. */
static const int servo_pins[FINGER_COUNT] = {
    PIN_SERVO_THUMB,
    PIN_SERVO_INDEX,
    PIN_SERVO_MIDDLE,
    PIN_SERVO_RING,
    PIN_SERVO_PINKY
};

/** @brief LEDC channel for each finger servo. */
static const ledc_channel_t servo_channels[FINGER_COUNT] = {
    LEDC_CH_THUMB,
    LEDC_CH_INDEX,
    LEDC_CH_MIDDLE,
    LEDC_CH_RING,
    LEDC_CH_PINKY
};

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

void servo_hal_init(void)
{
    /* Configure LEDC timer (shared by all servo channels) */
    ledc_timer_config_t timer_config = {
        .speed_mode      = SERVO_PWM_SPEED_MODE,
        .timer_num       = SERVO_PWM_TIMER,
        .duty_resolution = SERVO_PWM_RESOLUTION,
        .freq_hz         = SERVO_PWM_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_config));

    /* Configure each finger's LEDC channel */
    for (int i = 0; i < FINGER_COUNT; i++) {
        ledc_channel_config_t channel_config = {
            .speed_mode = SERVO_PWM_SPEED_MODE,
            .channel    = servo_channels[i],
            .timer_sel  = SERVO_PWM_TIMER,
            .intr_type  = LEDC_INTR_DISABLE,
            .gpio_num   = servo_pins[i],
            .duty       = servo_hal_degrees_to_duty(90),  /* Start extended (open) */
            .hpoint     = 0
        };
        ESP_ERROR_CHECK(ledc_channel_config(&channel_config));
    }
}

void servo_hal_set_duty(finger_t finger, uint32_t duty)
{
    if (finger >= FINGER_COUNT) {
        return;
    }

    ledc_set_duty(SERVO_PWM_SPEED_MODE, servo_channels[finger], duty);
    ledc_update_duty(SERVO_PWM_SPEED_MODE, servo_channels[finger]);
}

uint32_t servo_hal_degrees_to_duty(float degrees)
{
    /* Clamp to valid range */
    if (degrees < 0.0f) {
        degrees = 0.0f;
    } else if (degrees > 180.0f) {
        degrees = 180.0f;
    }

    /* Linear interpolation: duty = min + (degrees/180) * (max - min) */
    float duty = SERVO_DUTY_MIN + (degrees / 180.0f) * (SERVO_DUTY_MAX - SERVO_DUTY_MIN);
    return (uint32_t)duty;
}
