/**
 * @file servo_hal.c
 * @brief PCA9685-based servo HAL via I2C (ESP-IDF i2c_master driver).
 */

#include "servo_hal.h"
#include "driver/i2c_master.h"
#include "esp_err.h"
#include "esp_log.h"
#include "rom/ets_sys.h"
#include <freertos/FreeRTOS.h>

static const char *TAG = "servo_hal";

/*******************************************************************************
 * PCA9685 register map
 ******************************************************************************/

#define PCA9685_REG_MODE1         0x00
#define PCA9685_REG_MODE2         0x01
#define PCA9685_REG_LED0_ON_L     0x06
#define PCA9685_REG_PRE_SCALE     0xFE

#define PCA9685_MODE1_AI          (1 << 5)
#define PCA9685_MODE1_SLEEP       (1 << 4)
#define PCA9685_MODE2_OUTDRV      (1 << 2)

/*******************************************************************************
 * Module state
 ******************************************************************************/

static i2c_master_bus_handle_t s_i2c_bus = NULL;
static i2c_master_dev_handle_t s_pca_dev = NULL;

static uint8_t s_reg_buf[2];

/*******************************************************************************
 * Low-level I2C helpers
 ******************************************************************************/

static esp_err_t pca9685_write_reg(uint8_t reg, uint8_t val)
{
    s_reg_buf[0] = reg;
    s_reg_buf[1] = val;
    return i2c_master_transmit(s_pca_dev, s_reg_buf, 2, pdMS_TO_TICKS(50));
}

static esp_err_t pca9685_read_reg(uint8_t reg, uint8_t *out)
{
    return i2c_master_transmit_receive(s_pca_dev, &reg, 1, out, 1,
                                       pdMS_TO_TICKS(50));
}

static esp_err_t pca9685_set_pwm(uint8_t channel, uint16_t on, uint16_t off)
{
    uint8_t base = PCA9685_REG_LED0_ON_L + 4 * channel;
    esp_err_t ret;
    ret = pca9685_write_reg(base + 0, (uint8_t)(on  & 0xFF));
    if (ret != ESP_OK) return ret;
    ret = pca9685_write_reg(base + 1, (uint8_t)((on  >> 8) & 0x0F));
    if (ret != ESP_OK) return ret;
    ret = pca9685_write_reg(base + 2, (uint8_t)(off & 0xFF));
    if (ret != ESP_OK) return ret;
    return  pca9685_write_reg(base + 3, (uint8_t)((off >> 8) & 0x0F));
}

/*******************************************************************************
 * PCA9685 init
 ******************************************************************************/

static void pca9685_set_pwm_freq(uint32_t freq_hz)
{
    uint32_t prescale_val = (PCA9685_OSC_HZ + (4096UL * freq_hz) / 2)
                          / (4096UL * freq_hz) - 1;
    if (prescale_val < 3)   prescale_val = 3;
    if (prescale_val > 255) prescale_val = 255;

    uint8_t mode1 = 0;
    ESP_ERROR_CHECK(pca9685_read_reg(PCA9685_REG_MODE1, &mode1));
    ESP_ERROR_CHECK(pca9685_write_reg(PCA9685_REG_MODE1,
                                      (mode1 | PCA9685_MODE1_SLEEP)));
    ESP_ERROR_CHECK(pca9685_write_reg(PCA9685_REG_PRE_SCALE,
                                      (uint8_t)prescale_val));
    ESP_ERROR_CHECK(pca9685_write_reg(PCA9685_REG_MODE1,
                                      (mode1 & ~PCA9685_MODE1_SLEEP)));
    ets_delay_us(500);
}

static void i2c_bus_init(void)
{
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port             = I2C_PORT_NUM,
        .sda_io_num           = PIN_I2C_SDA,
        .scl_io_num           = PIN_I2C_SCL,
        .clk_source           = I2C_CLK_SRC_XTAL,
        .glitch_ignore_cnt    = 7,
        .flags.enable_internal_pullup = false,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &s_i2c_bus));

    uint8_t detected_addr = 0;
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        if (i2c_master_probe(s_i2c_bus, addr, pdMS_TO_TICKS(10)) == ESP_OK) {
            detected_addr = addr;
            break;
        }
    }
    if (detected_addr == 0) {
        ESP_LOGE(TAG, "No I2C device found — check wiring");
        return;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = detected_addr,
        .scl_speed_hz    = I2C_FREQ_HZ,
    };
    ESP_ERROR_CHECK(i2c_master_bus_add_device(s_i2c_bus, &dev_cfg, &s_pca_dev));
}

/*******************************************************************************
 * Public API
 ******************************************************************************/

void servo_hal_init(void)
{
    i2c_bus_init();

    ESP_ERROR_CHECK(pca9685_write_reg(PCA9685_REG_MODE1, PCA9685_MODE1_AI));
    ESP_ERROR_CHECK(pca9685_write_reg(PCA9685_REG_MODE2, PCA9685_MODE2_OUTDRV));
    pca9685_set_pwm_freq(SERVO_PWM_FREQ_HZ);

    uint32_t mid = servo_hal_degrees_to_duty(90.0f);
    const uint8_t parked[] = {
        PCA_CH_THUMB, PCA_CH_INDEX, PCA_CH_MIDDLE, PCA_CH_RING,
        PCA_CH_PINKY, PCA_CH_WRIST, PCA_CH_BICEP
    };
    for (size_t i = 0; i < sizeof(parked); i++) {
        servo_hal_set_duty(parked[i], mid);
    }
}

void servo_hal_set_duty(uint8_t channel, uint32_t duty)
{
    if (channel > 15) return;
    if (duty > 4095)  duty = 4095;

    esp_err_t err = pca9685_set_pwm(channel, 0, (uint16_t)duty);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "set_pwm ch=%u err=0x%x", channel, err);
    }
}

uint32_t servo_hal_degrees_to_duty(float degrees)
{
    if (degrees < 0.0f)        degrees = 0.0f;
    else if (degrees > 180.0f) degrees = 180.0f;

    float duty = SERVO_DUTY_MIN + (degrees / 180.0f) * (SERVO_DUTY_MAX - SERVO_DUTY_MIN);
    return (uint32_t)duty;
}
