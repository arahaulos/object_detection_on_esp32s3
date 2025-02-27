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
} detected_bbox;




void init_yolo(void);
detected_bbox* run_detector(uint8_t*fb, int32_t w, int32_t h, uint32_t *num_of_bboxes);

#ifdef __cplusplus
}
#endif