#pragma once

#include <stdint.h>
#include "yolo_inference.h"

typedef struct
{
    char ip[32];
    int port;
} server_info;


void init_server_info(server_info *addr, const char *ip, int port);

void send_image(server_info *addr, uint8_t *image_data, uint32_t image_data_len);
void send_bboxes(server_info *addr, detected_bbox *bboxes, uint32_t num_of_bboxes);
uint32_t request_object_detection(server_info *addr, detected_bbox *bboxes_buffer, uint32_t max_bboxes);