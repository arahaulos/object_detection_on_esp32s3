#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "wifi.h"
#include "camera.h"

#include "yolo_inference.h"
#include "client.h"


#include "wifi_ssid_and_pwd.h"


void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK( ret );

    connect_wifi(WIFI_SSID, WIFI_PWD);

    init_yolo();
    init_camera();


    server_info server;
    init_server_info(&server, "192.168.1.101", 6969);

    while (true) {
        uint32_t len, w, h;
        uint8_t *data = take_picture(&len, &w, &h);

        uint8_t *decoded_image = decode_image(data, len);

        uint32_t num_of_bboxes;
        detected_bbox *bboxes = run_detector(data, w, h, &num_of_bboxes);
        
        if (is_wifi_connected()) {
            send_image(&server, data, len);
            send_bboxes(&server, bboxes, num_of_bboxes);
        }

        vTaskDelay(10);
    }
}