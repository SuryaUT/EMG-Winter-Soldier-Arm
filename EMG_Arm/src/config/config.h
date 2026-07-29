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

/* Mode selectors must be #define (not enum) so they are visible to the
 * preprocessor. The `#if MAIN_MODE == PARITY_DUMP` guards in main.c compare
 * these at preprocess time; enum constants would both expand to 0 there,
 * making every such guard always-true. */
#define REAL_MAIN              0
#define EMG_MAIN               1
#define SERVO_CALIBRATOR_ANGLE 2
#define SERVO_CALIBRATOR_DUTY  3
#define GESTURE_TESTER         4
#define EMG_STANDALONE         5
#define CONTINUOUS_TEST        6
#define PARITY_DUMP            7

#define MAIN_MODE REAL_MAIN

/* Live-inference subsystem enables (independent). When 1, the live-inference
 * loops compile and run that subsystem; when 0 it is not compiled in.
 *   ENABLE_HAND  — hand gesture inference + servo control (fingers/wrist)
 *   ENABLE_BICEP — bicep flex detection + bicep servo control (PCA ch 6) */
#define ENABLE_HAND   0
#define ENABLE_BICEP  1

/* Bicep control style (only meaningful when ENABLE_BICEP == 1):
 *   1 — PROPORTIONAL: servo angle tracks muscle effort continuously, so a
 *       half-flex holds the arm halfway. Needs a two-point calibration
 *       (rest RMS + max-flex RMS). One EMG channel (ch3) is sufficient.
 *   0 — BINARY: original flex/rest threshold detector (single rest calib). */
#define BICEP_PROPORTIONAL 1

/* PARITY_DUMP: number of inference hops to dump.
 * The rep                                                                                                                                         ay is 75564 samples at hop 25 -> ~3022 hops. 3100 covers the whole
 * session, so the dump yields a full-session on-device accuracy figure. */
#define PARITY_DUMP_HOPS 3100

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
