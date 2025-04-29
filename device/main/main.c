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

#define FRAMETIME_BUFFER_SIZE 512

struct image_transmit_buffer_t
{
    uint8_t *image_buffer;
    bbox *bboxes_buffer;
    uint32_t num_of_bboxes;
    uint32_t image_len;
};

struct camera_image_buffer_t
{
    uint8_t *jpeg_image;
    uint8_t *decoded_image;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t image_len;
};


bool wifi_sent_flag = true;
bool wifi_transmit_flag = false;

bool camera_trigger_flag = false;
bool camera_image_ready_flag = false;


bool transmit_image_and_bboxes = false;
bool use_offboard_detection = false;


struct camera_image_buffer_t camera_image_buffer;
struct image_transmit_buffer_t image_transmit_buffer;
server_info server;

void init_image_transmit_buffer(struct image_transmit_buffer_t *buffer)
{
    buffer->image_buffer = (uint8_t*)heap_caps_malloc(IMAGE_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buffer->image_buffer) {
        printf("image buffer allocation failed\n");
    }
    buffer->bboxes_buffer = (bbox*)heap_caps_malloc(BBOX_BUFFER_SIZE*sizeof(bbox), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buffer->bboxes_buffer) {
        printf("bboxes buffer allocation failed\n");
    }
    buffer->num_of_bboxes = 0;
    buffer->image_len = 0;
}



void wifi_task(void *pvParameters)
{
    printf("Wifi task started\n");

    while (1) {
        if (wifi_transmit_flag && is_wifi_connected()) {
            wifi_transmit_flag = false;

            uint64_t start = esp_timer_get_time();

            send_image(&server, image_transmit_buffer.image_buffer, image_transmit_buffer.image_len);
            send_bboxes(&server, image_transmit_buffer.bboxes_buffer, image_transmit_buffer.num_of_bboxes);

            uint64_t end = esp_timer_get_time();
            //printf("Transmit time: %ld ms\n", (int32_t)((end - start)/1000));

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

            camera_image_buffer.jpeg_image = take_picture(&camera_image_buffer.image_len, &camera_image_buffer.image_width, &camera_image_buffer.image_height);

            uint64_t end = esp_timer_get_time();

            //printf("Taking picture: %ld ms\n", (int32_t)((end - start)/1000));

            start = esp_timer_get_time();

            camera_image_buffer.decoded_image = decode_image(camera_image_buffer.jpeg_image, camera_image_buffer.image_len);

            end = esp_timer_get_time();
            //printf("Decoding time: %ld ms\n", (int32_t)((end - start)/1000));

            camera_image_ready_flag = true;
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}


void object_detection_task(void *pvParameters)
{
    printf("Detection task started\n");

    int state = ODT_STATE_SYNC_CAM;


    uint8_t *local_image_buffer = (uint8_t *)heap_caps_malloc(IMAGE_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *local_decoded_image_buffer = NULL;
    uint32_t local_image_len = 0;
    uint32_t local_image_width = 0;
    uint32_t local_image_height = 0;

    bbox *bboxes = (bbox*)heap_caps_malloc(BBOX_BUFFER_SIZE*sizeof(bbox), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);;
    uint32_t num_of_bboxes = 0;



    if (transmit_image_and_bboxes || use_offboard_detection) {
        while (!test_connection(&server)) {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }



    uint32_t num_of_frames = 0;
    uint64_t start_ticks = esp_timer_get_time();

    while (1) {

        if (state == ODT_STATE_SYNC_CAM) {
            if (camera_image_ready_flag) {
                camera_image_ready_flag = false;

                //Copy JPEG image and decoded image to local buffers so that camera task can take next image
                if (local_decoded_image_buffer == NULL) {
                    local_decoded_image_buffer = (uint8_t *) heap_caps_malloc(camera_image_buffer.image_width*camera_image_buffer.image_height*3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
                }
                memcpy(local_decoded_image_buffer, camera_image_buffer.decoded_image, camera_image_buffer.image_width*camera_image_buffer.image_height*3);
                memcpy(local_image_buffer, camera_image_buffer.jpeg_image, camera_image_buffer.image_len);
                local_image_len = camera_image_buffer.image_len;
                local_image_width = camera_image_buffer.image_width;
                local_image_height = camera_image_buffer.image_height;

                camera_trigger_flag = true;

                state = ODT_STATE_RUN;
            }
        }
        if (state == ODT_STATE_RUN) {
            if (use_offboard_detection) {
                
                send_image(&server, local_image_buffer, local_image_len);
                num_of_bboxes = request_object_detection(&server, bboxes, BBOX_BUFFER_SIZE);

                printf("Received %ld bboxes\n", num_of_bboxes);

                state = ODT_STATE_SYNC_CAM;
            } else {
                uint64_t start = esp_timer_get_time();
                
                num_of_bboxes = run_detector(local_decoded_image_buffer, local_image_width, local_image_height, bboxes, BBOX_BUFFER_SIZE);

                uint64_t end = esp_timer_get_time();

                //printf("Detection time: %ld ms\n", (int32_t)((end - start)/1000));

                //If transmit_image_and_bboxes flag is set, image and bboxes are send to backend for debugging purposes
                if (transmit_image_and_bboxes) {
                    state = ODT_STATE_SYNC_WIFI;
                } else {
                    state = ODT_STATE_SYNC_CAM;
                }
            }

            num_of_frames++;
            printf("Avg frame time: %ld ms\n", (int32_t)((esp_timer_get_time() - start_ticks)/(1000*num_of_frames)));
        }
        if (state == ODT_STATE_SYNC_WIFI) {
            if (wifi_sent_flag) {
                wifi_sent_flag = false;

                //Copy bboxes and image for wifi task
                memcpy((void*)image_transmit_buffer.image_buffer, local_image_buffer, local_image_len*sizeof(uint8_t));
                memcpy((void*)image_transmit_buffer.bboxes_buffer, bboxes, num_of_bboxes*sizeof(bbox));

                image_transmit_buffer.num_of_bboxes = num_of_bboxes;
                image_transmit_buffer.image_len = local_image_len;
                
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

    init_server_info(&server, "192.168.1.101", 6900);
    init_yolo();
    init_camera();

    init_image_transmit_buffer(&image_transmit_buffer);
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