# Bicep Actuation + ENABLE_HAND / ENABLE_BICEP Flags

**Date:** 2026-07-19
**Status:** Approved

## Goal

1. Make the bicep flex detector actually drive its servo: on **FLEX** command 140°, on **REST** command 60°, on PCA channel 6 (the 7th PWM port = `PCA_CH_BICEP` / `JOINT_BICEP`).
2. Add two independent compile-time flags, `ENABLE_HAND` and `ENABLE_BICEP`. When set to 1, live inference compiles and runs the corresponding subsystem (hand gesture inference/control, and/or bicep flex detection/control).

## Bicep angle range

- Min (unflexed / rest): **60°**
- Max (flexed): **140°**

## Design

### 1. Flags — `config.h`

Add next to the existing compile-time switches (`MAIN_MODE`, `LIVE_EMG`):

```c
#define ENABLE_HAND   1   /* compile + run hand gesture inference/control */
#define ENABLE_BICEP  1   /* compile + run bicep flex detection/control  */
```

Independent: either, both, or neither. Guarded with `#if ENABLE_HAND` / `#if ENABLE_BICEP`
so a disabled subsystem is not compiled in. Chosen over runtime bools because the
requirement is that inference "compiles and runs" per active flag.

### 2. Bicep actuation — new `bicep_apply()` in `bicep.c` / `bicep.h`

Centralizes the state→angle mapping and edge suppression in the bicep module so the
loop bodies don't duplicate it.

```c
#define BICEP_FLEX_ANGLE 140.0f   /* max flex */
#define BICEP_REST_ANGLE  60.0f   /* min / unflexed */

void bicep_apply(bicep_state_t state) {
    static int last_applied = -1;             /* forces a command on first call */
    if ((int)state == last_applied) return;   /* edge-triggered: only on change */
    last_applied = state;
    float angle = (state == BICEP_STATE_FLEX) ? BICEP_FLEX_ANGLE : BICEP_REST_ANGLE;
    hand_set_finger_angle(JOINT_BICEP, angle); /* JOINT_BICEP -> PCA ch 6 = 7th port */
}
```

- `hand_set_finger_angle(JOINT_BICEP, …)` routes to PCA channel 6 via the servo HAL and
  does not index the finger min/max tables, so `JOINT_BICEP` is safe.
- `bicep.c` gains an include of `drivers/hand.h`.
- Servo is commanded only on FLEX<->REST transitions (edge-triggered), matching the
  hand's `if (gesture_idx != last_gesture)` pattern; avoids ~40 redundant I2C writes/sec.

### 3. Restructure both live-inference loops

`run_inference_loop` (STATE_PREDICTING) and `run_standalone_loop` (EMG_STANDALONE).

Currently the bicep call sits inside `if (gesture_idx >= 0)`, coupling it to the hand.
New per-hop structure:

```c
#if ENABLE_HAND
    /* feature extraction + LDA/ensemble/MLP + vote_postprocess
       + gestures_execute + gesture JSON print */
#endif
#if ENABLE_BICEP
    bicep_apply(bicep_detect());   /* independent of hand */
#endif
```

- Model init (`n_models`, ensemble/MLP init, `last_gesture`) wrapped in `#if ENABLE_HAND`.
- `inference_add_sample` + hop-size gating stay **unguarded** — both subsystems need the
  filtered sample buffer (bicep reads ch3 RMS from it; no features required).
- The old discarded `bicep_state_t bicep = bicep_detect(); (void)bicep;` is removed.

## Net behavior

- `ENABLE_HAND=1, ENABLE_BICEP=1` → both run.
- Bicep-only → lightweight: skips all feature extraction/models, just RMS-threshold → servo.
- Hand-only → today's behavior minus the dead bicep call.

## Testing

- Compiles cleanly in all four flag combinations.
- On replay/live: flexing the bicep drives the ch6 servo to 140°, relaxing to 60°,
  commanded only on transitions.
