/**
 * @file gestures.c
 * @brief Named gesture implementation.
 *
 * Implements gesture functions using the hand driver.
 */

#include "gestures.h"
#include "drivers/hand.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>


/*******************************************************************************
 * Private Data
 ******************************************************************************/

/** @brief Gesture name lookup table. */
static const char* gesture_names[GESTURE_COUNT] = {
    "NONE",
    "REST",
    "FIST",
    "OPEN",
    "HOOK_EM",
    "THUMBS_UP"
};

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

void gestures_execute(gesture_t gesture)
{
    switch (gesture) {
        case GESTURE_REST:
            gesture_rest();
            break;
        case GESTURE_FIST:
            gesture_fist();
            break;
        case GESTURE_OPEN:
            gesture_open();
            break;
        case GESTURE_HOOK_EM:
            gesture_hook_em();
            break;
        case GESTURE_THUMBS_UP:
            gesture_thumbs_up();
            break;
        default:
            break;
    }
}

gesture_t parse_gesture(const char *s)
{
    if (strcmp(s, "rest") == 0)       return GESTURE_REST;
    if (strcmp(s, "fist") == 0)       return GESTURE_FIST;
    if (strcmp(s, "open") == 0)       return GESTURE_OPEN;
    if (strcmp(s, "hook-em") == 0 ||
        strcmp(s, "hookem") == 0)     return GESTURE_HOOK_EM;
    if (strcmp(s, "thumbs-up") == 0 ||
        strcmp(s, "thumbsup") == 0)   return GESTURE_THUMBS_UP;

    return GESTURE_NONE;
}

const char* gestures_get_name(gesture_t gesture)
{
    if (gesture >= GESTURE_COUNT) {
        return "UNKNOWN";
    }
    return gesture_names[gesture];
}

/*******************************************************************************
 * Individual Gesture Functions
 ******************************************************************************/

void gesture_open(void)
{
    hand_set_finger_angle(FINGER_THUMB, minAngles[FINGER_THUMB]);
    hand_set_finger_angle(FINGER_INDEX, minAngles[FINGER_INDEX]);
    hand_set_finger_angle(FINGER_MIDDLE, minAngles[FINGER_MIDDLE]);
    hand_set_finger_angle(FINGER_RING, minAngles[FINGER_RING]);
    hand_set_finger_angle(FINGER_PINKY, minAngles[FINGER_PINKY]);
}

void gesture_fist(void)
{
    hand_set_finger_angle(FINGER_INDEX, maxAngles[FINGER_INDEX]);
    hand_set_finger_angle(FINGER_MIDDLE, maxAngles[FINGER_MIDDLE]);
    hand_set_finger_angle(FINGER_RING, maxAngles[FINGER_RING]);
    hand_set_finger_angle(FINGER_PINKY, maxAngles[FINGER_PINKY]);
    hand_set_finger_angle(FINGER_THUMB, maxAngles[FINGER_THUMB]);
}

void gesture_hook_em(void)
{
    /* Index and pinky extended, others flexed */
    hand_set_finger_angle(FINGER_THUMB, maxAngles[FINGER_THUMB]);
    hand_set_finger_angle(FINGER_INDEX, minAngles[FINGER_INDEX]);
    hand_set_finger_angle(FINGER_MIDDLE, maxAngles[FINGER_MIDDLE]);
    hand_set_finger_angle(FINGER_RING, maxAngles[FINGER_RING]);
    hand_set_finger_angle(FINGER_PINKY, minAngles[FINGER_PINKY]);
}

void gesture_thumbs_up(void)
{
    /* Thumb extended, others flexed */
    hand_set_finger_angle(FINGER_THUMB, minAngles[FINGER_THUMB]);
    hand_set_finger_angle(FINGER_INDEX, maxAngles[FINGER_INDEX]);
    hand_set_finger_angle(FINGER_MIDDLE, maxAngles[FINGER_MIDDLE]);
    hand_set_finger_angle(FINGER_RING, maxAngles[FINGER_RING]);
    hand_set_finger_angle(FINGER_PINKY, maxAngles[FINGER_PINKY]);
}

void gesture_rest(void)
{
    hand_set_finger_angle(FINGER_THUMB, (maxAngles[FINGER_THUMB] + minAngles[FINGER_THUMB])/2);
    hand_set_finger_angle(FINGER_INDEX, (maxAngles[FINGER_INDEX] + minAngles[FINGER_INDEX])/2);
    hand_set_finger_angle(FINGER_MIDDLE, (maxAngles[FINGER_MIDDLE] + minAngles[FINGER_MIDDLE])/2);
    hand_set_finger_angle(FINGER_RING, (maxAngles[FINGER_RING] + minAngles[FINGER_RING])/2);
    hand_set_finger_angle(FINGER_PINKY, (maxAngles[FINGER_PINKY] + minAngles[FINGER_PINKY])/2);
}

/*******************************************************************************
 * Demo Functions
 ******************************************************************************/

void gestures_demo_fingers(uint32_t delay_ms)
{
    for (int finger = 0; finger < FINGER_COUNT; finger++) {
        hand_flex_finger((finger_t)finger);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
        hand_unflex_finger((finger_t)finger);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
}

void gestures_demo_fist(uint32_t delay_ms)
{
    gesture_fist();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    gesture_open();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
}
