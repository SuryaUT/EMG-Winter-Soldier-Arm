/**
 * @file hand.c
 * @brief Hand driver implementation.
 *
 * Provides finger-level control using the servo HAL.
 */

#include "hand.h"
#include "hal/servo_hal.h"

float maxAngles[] = {155, 155, 180, 165, 150};
float minAngles[] = {65, 45, 45, 30, 25};

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

void hand_init(void)
{
    servo_hal_init();
}

void hand_flex_finger(finger_t finger)
{
    servo_hal_set_duty(finger, SERVO_DUTY_MAX);
}

void hand_unflex_finger(finger_t finger)
{
    servo_hal_set_duty(finger, SERVO_DUTY_MIN);
}

void hand_set_finger_angle(finger_t finger, float degrees)
{
    uint32_t duty = servo_hal_degrees_to_duty(degrees);
    servo_hal_set_duty(finger, duty);
}

void hand_flex_all(void)
{
    for (int i = 0; i < FINGER_COUNT; i++) {
        hand_flex_finger((finger_t)i);
    }
}

void hand_unflex_all(void)
{
    for (int i = 0; i < FINGER_COUNT; i++) {
        hand_unflex_finger((finger_t)i);
    }
}
