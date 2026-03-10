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

extern float maxAngles[];
extern float minAngles[];
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
void hand_flex_finger(finger_t finger);

/**
 * @brief Unflex a finger (extend/open it).
 *
 * @param finger Which finger to unflex
 */
void hand_unflex_finger(finger_t finger);

/**
 * @brief Set a finger to a specific angle.
 *
 * @param finger  Which finger to move
 * @param degrees Angle (0 = extended, 180 = fully flexed)
 */
void hand_set_finger_angle(finger_t finger, float degrees);

/**
 * @brief Flex all fingers at once.
 */
void hand_flex_all(void);

/**
 * @brief Unflex all fingers at once.
 */
void hand_unflex_all(void);

#endif /* HAND_H */
