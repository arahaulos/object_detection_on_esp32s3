#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "wifi.h"
#include "camera.h"

#include "yolo_inference.h"
#include "client.h"

#include "wifi_ssid_and_pwd.h"

#define ODT_STATE_SYNC_WIFI 0
#define ODT_STATE_SYNC_CAM 1
#define ODT_STATE_RUN 2

#define IMAGE_BUFFER_SIZE 16*1024
#define BBOX_BUFFER_SIZE 64


uint8_t *wifi_image_buffer;
detected_bbox *wifi_bboxes_buffer;
uint32_t wifi_num_of_bboxes;
uint32_t wifi_image_len;

bool wifi_sent_flag = true;
bool wifi_transmit_flag = false;



uint8_t *camera_image;
uint8_t *decoded_image;
uint32_t image_width;
uint32_t image_height;
uint32_t image_len;

bool camera_trigger_flag = false;
bool camera_image_ready_flag = false;


void wifi_task(void *pvParameters)
{
    printf("Wifi task started\n");

    server_info server;
    init_server_info(&server, "192.168.1.101", 6969);

    while (1) {
        if (wifi_transmit_flag && is_wifi_connected()) {
            wifi_transmit_flag = false;

            uint64_t start = esp_timer_get_time();

            send_image(&server, wifi_image_buffer, wifi_image_len);
            send_bboxes(&server, wifi_bboxes_buffer, wifi_num_of_bboxes);

            uint64_t end = esp_timer_get_time();
            printf("Transmit time: %ld ms\n", (int32_t)((end - start)/1000));

            wifi_sent_flag = true;
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}


void camera_task(void *pvParametrs)
{
    printf("Camera task started\n");

    while (1) {
        if (camera_trigger_flag) {
            camera_trigger_flag = false;

            uint64_t start = esp_timer_get_time();

            camera_image = take_picture(&image_len, &image_width, &image_height);
            decoded_image = decode_image(camera_image, image_len);

            uint64_t end = esp_timer_get_time();
            printf("Camera time: %ld ms\n", (int32_t)((end - start)/1000));

            camera_image_ready_flag = true;
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}


void object_detection_task(void *pvParameters)
{
    printf("Detection task started\n");

    int state = ODT_STATE_SYNC_CAM;


    uint8_t *local_image_buffer = (uint8_t *) heap_caps_malloc(IMAGE_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *local_decoded_image_buffer = NULL;
    uint32_t local_image_len = 0;
    uint32_t local_image_width = 0;
    uint32_t local_image_height = 0;

    detected_bbox *bboxes = NULL;
    uint32_t num_of_bboxes = 0;

    while (1) {

        if (state == ODT_STATE_SYNC_CAM) {
            if (camera_image_ready_flag) {
                camera_image_ready_flag = false;

                if (local_decoded_image_buffer == NULL) {
                    local_decoded_image_buffer = (uint8_t *) heap_caps_malloc(image_width*image_height*3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
                }

                memcpy(local_decoded_image_buffer, decoded_image, image_width*image_height*3);
                memcpy(local_image_buffer, camera_image, image_len);
                local_image_len = image_len;
                local_image_width = image_width;
                local_image_height = image_height;

                camera_trigger_flag = true;

                state = ODT_STATE_RUN;
            }
        }
        if (state == ODT_STATE_RUN) {

            uint64_t start = esp_timer_get_time();
            
            bboxes = run_detector(local_decoded_image_buffer, local_image_width, local_image_height, &num_of_bboxes);

            uint64_t end = esp_timer_get_time();

            printf("Detection time: %ld ms\n", (int32_t)((end - start)/1000));

            state = ODT_STATE_SYNC_WIFI;

        }
        if (state == ODT_STATE_SYNC_WIFI) {
            if (wifi_sent_flag) {
                wifi_sent_flag = false;

                memcpy((void*)wifi_image_buffer, local_image_buffer, local_image_len*sizeof(uint8_t));
                memcpy((void*)wifi_bboxes_buffer, bboxes, num_of_bboxes*sizeof(detected_bbox));

                wifi_num_of_bboxes = num_of_bboxes;
                wifi_image_len = local_image_len;
                
                wifi_transmit_flag = true;

                state = ODT_STATE_SYNC_CAM;
            }
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}




void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK( ret );

    esp_netif_init();
    esp_event_loop_create_default();

    connect_wifi(WIFI_SSID, WIFI_PWD);

    init_yolo();
    init_camera();

    wifi_image_buffer = (uint8_t *) heap_caps_malloc(IMAGE_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!wifi_image_buffer) {
        printf("image buffer allocation failed\n");
    }
    wifi_bboxes_buffer = (detected_bbox *) heap_caps_malloc(BBOX_BUFFER_SIZE*sizeof(detected_bbox), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!wifi_bboxes_buffer) {
        printf("bboxes buffer allocation failed\n");
    }
    wifi_num_of_bboxes = 0;
    wifi_image_len = 0;

    wifi_sent_flag = true;
    wifi_transmit_flag = false;

    camera_image_ready_flag = false;
    camera_trigger_flag = true;

    printf("Creating object detection task\n");
    xTaskCreate(object_detection_task, "object_detection_task", 4096, NULL, 1, NULL);

    printf("Creating wifi task\n");
    xTaskCreate(wifi_task, "wifi_task", 4096, NULL, 1, NULL);

    printf("Creating camera task\n");
    xTaskCreate(camera_task, "cam_task", 4096, NULL, 1, NULL);

}