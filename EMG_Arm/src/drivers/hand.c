/**
 * @file hand.c
 * @brief Hand driver implementation.
 *
 * Provides finger-level control using the servo HAL.
 */

#include "hand.h"
#include "hal/servo_hal.h"

/* Per-joint angle limits — index matches joint_t enum order.
 * Tune these so servos don't fight their mechanical stops. */
float maxAngles[FINGER_JOINT_COUNT + 1] = {
    155,  /* JOINT_THUMB  */
    155,  /* JOINT_INDEX  */
    180,  /* JOINT_MIDDLE */
    165,  /* JOINT_RING   */
    150,  /* JOINT_PINKY  */
    135,  /* JOINT_WRIST  */
};
float minAngles[FINGER_JOINT_COUNT + 1] = {
    65,   /* JOINT_THUMB  */
    45,   /* JOINT_INDEX  */
    45,   /* JOINT_MIDDLE */
    30,   /* JOINT_RING   */
    25,   /* JOINT_PINKY  */
    45,   /* JOINT_WRIST  */
};

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

void hand_init(void)
{
    servo_hal_init();
}

void hand_flex_finger(joint_t joint)
{
    hand_set_finger_angle(joint, maxAngles[joint]);
}

void hand_unflex_finger(joint_t joint)
{
    hand_set_finger_angle(joint, minAngles[joint]);
}

void hand_set_finger_angle(joint_t joint, float degrees)
{
    uint32_t duty = servo_hal_degrees_to_duty(degrees);
    servo_hal_set_duty(joint, duty);
}

void hand_set_finger_duty(joint_t joint, float duty)
{
    servo_hal_set_duty(joint, duty);
}

void hand_flex_all(void)
{
    for (int i = 0; i < FINGER_JOINT_COUNT; i++) {
        hand_flex_finger((joint_t)i);
    }
}

void hand_unflex_all(void)
{
    for (int i = 0; i < FINGER_JOINT_COUNT; i++) {
        hand_unflex_finger((joint_t)i);
    }
}
