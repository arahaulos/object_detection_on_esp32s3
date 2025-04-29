#pragma once


#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct
{
    uint32_t object_type;
    float confidence;
    float estimated_distance;
    float x;
    float y;
    float w;
    float h;
} bbox;




void init_yolo(void);
int run_detector(uint8_t*fb, int32_t w, int32_t h, bbox *bbox_buffer, int buffer_size);

#ifdef __cplusplus
}
#endif