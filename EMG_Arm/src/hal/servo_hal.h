/**
 * @file servo_hal.h
 * @brief Hardware Abstraction Layer for servo PWM control via PCA9685.
 *
 * The ESP32 talks I2C to a PCA9685 16-channel 12-bit PWM controller, which
 * generates the servo pulses. This module wraps the I2C transactions and
 * presents a simple "set channel duty" interface.
 *
 * @note This is Layer 1 (HAL). Only drivers/ should use this directly.
 */

#ifndef SERVO_HAL_H
#define SERVO_HAL_H

#include <stdint.h>
#include "config/config.h"

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

/**
 * @brief Initialize the I2C bus and the PCA9685 controller.
 *
 * Configures the I2C master, resets the PCA9685, sets the PWM frequency
 * to SERVO_PWM_FREQ_HZ, and parks every used channel at 90°.
 *
 * @note Must be called once before any other servo_hal functions.
 */
void servo_hal_init(void);

/**
 * @brief Set the PWM duty for one PCA9685 channel.
 *
 * @param channel  PCA9685 channel index (0..15). Use PCA_CH_* macros.
 * @param duty     12-bit OFF tick count (SERVO_DUTY_MIN..SERVO_DUTY_MAX
 *                 for typical servos).
 */
void servo_hal_set_duty(uint8_t channel, uint32_t duty);

/**
 * @brief Convert degrees to a PCA9685 12-bit duty value.
 *
 * @param degrees Angle in degrees (0..180).
 * @return Corresponding duty tick count (clamped to the servo range).
 */
uint32_t servo_hal_degrees_to_duty(float degrees);

#endif /* SERVO_HAL_H */
