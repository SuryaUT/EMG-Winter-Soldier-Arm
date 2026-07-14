/**
 * @file gestures.h
 * @brief Named gesture definitions and execution.
 *
 * This module provides named gestures (fist, open, hook_em, etc.)
 * that map to specific finger configurations. These are the gestures
 * that the ML model will predict.
 *
 * @note This is Layer 3 (Core). Uses drivers/hand internally.
 */

#ifndef GESTURES_H
#define GESTURES_H

#include <stdint.h>
#include <string.h>
#include "config/config.h"

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

/**
 * @brief Execute a gesture by its enum value.
 *
 * @param gesture Which gesture to perform
 */
void gestures_execute(gesture_t gesture);

/* NOTE: there is deliberately no index->gesture_t helper. Class indices are
 * ALPHABETICAL (fist, hook_em, open, rest, thumbs_up) while gesture_t is not, so
 * any (idx + 1) style mapping is wrong. Convert by NAME via
 * inference_get_gesture_enum() / inference_get_gesture_by_name() instead. */

gesture_t parse_gesture(const char *s);

/**
 * @brief Get the name of a gesture as a string.
 *
 * @param gesture Which gesture
 * @return Pointer to gesture name string (e.g., "FIST", "OPEN")
 */
const char* gestures_get_name(gesture_t gesture);

/*******************************************************************************
 * Individual Gesture Functions
 ******************************************************************************/

/** @brief Open hand - all fingers extended. */
void gesture_open(void);

/** @brief Fist - all fingers flexed. */
void gesture_fist(void);

/** @brief Hook 'em Horns - index and pinky extended, others flexed. */
void gesture_hook_em(void);

/** @brief Thumbs up - thumb extended, others flexed. */
void gesture_thumbs_up(void);

/** @brief Rest - same as open (neutral position). */
void gesture_rest(void);

/*******************************************************************************
 * Demo Functions
 ******************************************************************************/

/**
 * @brief Demo: cycle each finger individually.
 *
 * @param delay_ms Milliseconds between movements
 */
void gestures_demo_fingers(uint32_t delay_ms);

/**
 * @brief Demo: open and close fist repeatedly.
 *
 * @param delay_ms Milliseconds between open/close
 */
void gestures_demo_fist(uint32_t delay_ms);

#endif /* GESTURES_H */
