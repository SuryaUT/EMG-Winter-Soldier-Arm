# CMakeLists.txt for Bucky Arm EMG Robotic Hand
#
# Project structure:
#   src/
#   ├── config/     Configuration headers
#   ├── hal/        Hardware Abstraction Layer
#   ├── drivers/    Device drivers
#   ├── core/       Core business logic
#   └── app/        Application entry point

# Collect all source files from each layer
set(HAL_SOURCES
    hal/servo_hal.c
)

set(DRIVER_SOURCES
    drivers/hand.c
    drivers/emg_sensor.c
)

set(CORE_SOURCES
    core/bicep.c
    core/calibration.c
    core/gestures.c
    core/inference.c
    core/inference_ensemble.c
    core/inference_mlp.cc
    core/emg_model_data.cc
)

set(APP_SOURCES
    app/main.c
)

# Register component with ESP-IDF
idf_component_register(
    SRCS
        ${HAL_SOURCES}
        ${DRIVER_SOURCES}
        ${CORE_SOURCES}
        ${APP_SOURCES}
    INCLUDE_DIRS
        .
    REQUIRES esp_adc nvs_flash esp-dsp esp-tflite-micro
)
