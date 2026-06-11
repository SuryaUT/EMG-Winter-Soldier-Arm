/**
 * @file hand.h
 * @brief Hand driver for individual finger control.
 *
 * This module provides an intuitive interface for controlling
 * individual fingers - flex, unflex, or set to a specific angle.
 *
 * @note This is Layer 2 (Driver). Uses hal/servo_hal internally.
 */

#ifndef HAND_H
#define HAND_H

#include "config/config.h"

/* Joints managed by this driver: thumb through wrist (excludes bicep). */
#define FINGER_JOINT_COUNT  (JOINT_PINKY + 1)

extern float maxAngles[FINGER_JOINT_COUNT + 1];
extern float minAngles[FINGER_JOINT_COUNT + 1];
/*******************************************************************************
 * Public Functions
 ******************************************************************************/

/**
 * @brief Initialize the hand (all finger servos).
 *
 * Sets up PWM for all servos. All fingers start extended (open).
 */
void hand_init(void);

/**
 * @brief Flex a finger (close it).
 *
 * @param finger Which finger to flex
 */
void hand_flex_finger(joint_t finger);

/**
 * @brief Unflex a finger (extend/open it).
 *
 * @param finger Which finger to unflex
 */
void hand_unflex_finger(joint_t finger);

/**
 * @brief Set a finger to a specific angle.
 *
 * @param finger  Which finger to move
 * @param degrees Angle (0 = extended, 180 = fully flexed)
 */
void hand_set_finger_angle(joint_t finger, float degrees);

/**
 * @brief Set a finger servo to a specific duty cycle.
 *
 * @param finger  Which finger to move
 * @param duty Duty cycle in ticks
 */
void hand_set_finger_duty(joint_t finger, float duty);

/**
 * @brief Flex all fingers at once.
 */
void hand_flex_all(void);

/**
 * @brief Unflex all fingers at once.
 */
void hand_unflex_all(void);

#endif /* HAND_H */
