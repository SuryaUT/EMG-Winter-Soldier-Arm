/**
 * @file emg_sensor.h
 * @brief EMG sensor driver for reading muscle signals.
 *
 * This module provides EMG data acquisition from ADC channels connected
 * to MyoWare sensors. Outputs calibrated millivolt values (0-3300 mV).
 *
 * @note This is Layer 2 (Driver).
 */

#ifndef EMG_SENSOR_H
#define EMG_SENSOR_H

#include <stdint.h>
#include "config/config.h"

/*******************************************************************************
 * Data Types
 ******************************************************************************/

/**
 * @brief Single EMG reading from all channels.
 */
typedef struct {
    uint32_t timestamp_ms;              /**< Timestamp in milliseconds */
    uint16_t channels[EMG_NUM_CHANNELS]; /**< ADC values for each channel */
} emg_sample_t;

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

/**
 * @brief Initialize the EMG sensor system.
 *
 * Configures ADC channels and calibration for real sensor reading.
 */
void emg_sensor_init(void);

/**
 * @brief Read current values from all EMG channels.
 *
 * @param sample Pointer to struct to fill with current readings
 */
void emg_sensor_read(emg_sample_t *sample);

/**
 * @brief Get the current timestamp in milliseconds.
 *
 * @return Milliseconds since boot
 */
uint32_t emg_sensor_get_timestamp_ms(void);

#endif /* EMG_SENSOR_H */
