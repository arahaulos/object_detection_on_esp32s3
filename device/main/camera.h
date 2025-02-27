#pragma once

#include <stdint.h>

void init_camera();

uint8_t* take_picture(uint32_t *file_len_out, uint32_t *image_width, uint32_t *image_height);

uint8_t* decode_image(uint8_t *file, uint32_t file_len);