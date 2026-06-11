/**
 * @file config.h
 * @brief Centralized configuration for the EMG-controlled robotic hand.
 *
 * All hardware pin definitions and system constants in one place.
 * Modify this file to adapt to different hardware configurations.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include "driver/gpio.h"

/*******************************************************************************
 * Main Modes
 ******************************************************************************/

enum {REAL_MAIN, EMG_MAIN, SERVO_CALIBRATOR_ANGLE, SERVO_CALIBRATOR_DUTY, GESTURE_TESTER, EMG_STANDALONE, CONTINUOUS_TEST};

#define MAIN_MODE REAL_MAIN

/*******************************************************************************
 * I2C Configuration (PCA9685 servo controller bus)
 ******************************************************************************/

#define PIN_I2C_SDA               GPIO_NUM_8           /**< Adjust to your wiring */
#define PIN_I2C_SCL               GPIO_NUM_7           /**< Adjust to your wiring */
#define I2C_PORT_NUM              0                    /**< I2C peripheral 0 */
#define I2C_FREQ_HZ               400000               /**< 400 kHz fast-mode */

/*******************************************************************************
 * PCA9685 Configuration
 ******************************************************************************/

#define PCA9685_I2C_ADDR          0x40    /**< Default 7-bit address (A5..A0 = 0) */
#define PCA9685_OSC_HZ            25000000UL  /**< Internal oscillator: 25 MHz */

/*******************************************************************************
 * Servo PWM Configuration
 ******************************************************************************/

#define SERVO_PWM_FREQ_HZ         50    /**< Standard servo frequency */

/* 12-bit counter @ 50 Hz: 1 tick = 20 ms / 4096 ≈ 4.88 us.
 * ~540 us pulse (~0°): 540 / 4.88 ≈ 110
 * ~2490 us pulse (~180°): 2490 / 4.88 ≈ 510
 * Recalibrate per servo if needed. */
#define SERVO_DUTY_MIN            110   /**< PCA tick count for 0° (extended) */
#define SERVO_DUTY_MAX            510   /**< PCA tick count for 180° (flexed) */

/*******************************************************************************
 * PCA9685 Channel Assignments
 *
 * The PCA9685 has 16 channels (0..15). We use 7:
 *   Fingers on channels 0..4 (matches joint_t enum order — important).
 *   Wrist and bicep on 5 and 6.
 ******************************************************************************/

// #define PCA_CH_THUMB              0
// #define PCA_CH_INDEX              1
// #define PCA_CH_MIDDLE             2
// #define PCA_CH_RING               3
// #define PCA_CH_PINKY              4
// #define PCA_CH_WRIST              5
// #define PCA_CH_BICEP              6
enum {PCA_CH_THUMB, PCA_CH_INDEX, PCA_CH_MIDDLE, PCA_CH_RING, PCA_CH_PINKY, PCA_CH_WRIST, PCA_CH_BICEP};

/*******************************************************************************
 * EMG Configuration
 ******************************************************************************/

#define EMG_NUM_CHANNELS          4     /**< Number of EMG sensor channels */
#define EMG_SAMPLE_RATE_HZ        1000  /**< Samples per second per channel */

/*******************************************************************************
 * Common Type Definitions
 ******************************************************************************/

/**
 * @brief Finger identification.
 */
typedef enum {
    JOINT_THUMB = 0,
    JOINT_INDEX,
    JOINT_MIDDLE,
    JOINT_RING,
    JOINT_PINKY,
    JOINT_WRIST,
    JOINT_BICEP,
    JOINT_COUNT    /**< Total number of fingers (5) */
} joint_t;

/**
 * @brief Recognized gestures.
 */
typedef enum {
    GESTURE_NONE = 0,
    GESTURE_REST,
    GESTURE_FIST,
    GESTURE_OPEN,
    GESTURE_HOOK_EM,
    GESTURE_THUMBS_UP,
    GESTURE_COUNT
} gesture_t;

#endif /* CONFIG_H */
