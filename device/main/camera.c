#include "camera.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_jpg_decode.h"
#include "jpeg_decoder.h"

#include <esp_heap_caps.h>

#include "img_converters.h"

static const char *TAG = "CAMERA";

static camera_config_t camera_config = {
    .pin_pwdn = -1,
    .pin_reset = -1,
    .pin_xclk = 10,
    .pin_sccb_sda = 40,
    .pin_sccb_scl = 39,

    .pin_d7 = 48,
    .pin_d6 = 11,
    .pin_d5 = 12,
    .pin_d4 = 14,
    .pin_d3 = 16,
    .pin_d2 = 18,
    .pin_d1 = 17,
    .pin_d0 = 15,
    .pin_vsync = 38,
    .pin_href = 47,
    .pin_pclk = 13,

   
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .fb_location = CAMERA_FB_IN_PSRAM,

    .pixel_format = PIXFORMAT_JPEG, 
    .frame_size = FRAMESIZE_QVGA,     

    .jpeg_quality = 12, 
    .fb_count = 1,     
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

size_t decoded_framebuffer_size;
uint8_t *decoded_framebuffer = NULL;

void init_camera()
{
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera Init Failed");
    }

    decoded_framebuffer_size = 320*240*3;
    decoded_framebuffer = (uint8_t *) heap_caps_malloc(decoded_framebuffer_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

uint8_t* take_picture(uint32_t *file_len_out, uint32_t *image_width, uint32_t *image_height)
{
    static camera_fb_t *cam_fb = NULL;
    if (cam_fb != NULL) {
        esp_camera_fb_return(cam_fb);
    }
    cam_fb = esp_camera_fb_get();
    
    *file_len_out = cam_fb->len;
    *image_width = cam_fb->width;
    *image_height = cam_fb->height;

    return cam_fb->buf;
}

uint8_t* decode_image(uint8_t *file, uint32_t file_len)
{
    /*if (!fmt2rgb888(file, file_len, PIXFORMAT_JPEG, decoded_framebuffer)) {
        ESP_LOGE(TAG, "JPG decoding failed");
    }*/

    esp_jpeg_image_cfg_t jpeg_cfg = {
        .indata = (uint8_t *)file,
        .indata_size = file_len,
        .outbuf = decoded_framebuffer,
        .outbuf_size = decoded_framebuffer_size,
        .out_format = JPEG_IMAGE_FORMAT_RGB888,
        .out_scale = JPEG_IMAGE_SCALE_0,
        .flags = {
            .swap_color_bytes = 1,
        }
    };
    esp_jpeg_image_output_t outimg;
    
    esp_jpeg_decode(&jpeg_cfg, &outimg);

    return decoded_framebuffer;
}
