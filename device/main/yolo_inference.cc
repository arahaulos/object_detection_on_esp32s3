#include "yolo_inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <math.h>


#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_timer.h"

#define IMGZ_192
#define USE_YOLOV5N6_XIAO
#define USE_BILINEAR_INTERP 0

#ifdef USE_YOLOV5N6_XIAO

#ifdef IMGZ_192
#include "yolov5_192.h"
#endif

#ifdef IMGZ_256
#include "yolov5_256.h"
#endif

#endif

#ifdef USE_YOLOV8N

#ifdef IMGZ_192
#include "yolov8_192.h"
#endif

#ifdef IMGZ_256
#include "yolov8_256.h"
#endif


#endif

#include <esp_heap_caps.h>

constexpr float CONFIDENCE_TRESHOLD = 0.25f;
constexpr float IOU_TRESHOLD = 0.25f;
constexpr int NUM_OF_CLASSES = 1;

constexpr int MAX_TEMP_BBOX = 256;
bbox temp_bboxes[MAX_TEMP_BBOX];


constexpr float OBJECT_ESTIMATED_SIZE[NUM_OF_CLASSES*2] =
{
    0.6f, 1.75f //Person
};


float estimate_distance(bbox *bb, float vertical_fov = 66.0f*M_PI/180.0f, float aspect_ratio = 320.0f/240.0f)
{
    if (bb->object_type < 0 || bb->object_type >= NUM_OF_CLASSES) {
        return 0.0;
    }

    float horizontal_fov = vertical_fov / aspect_ratio;

    float object_width_angle = bb->w * vertical_fov;
    float object_height_angle = bb->h * horizontal_fov;

    float object_estimated_width = OBJECT_ESTIMATED_SIZE[bb->object_type*2 + 0];
    float object_estimated_height = OBJECT_ESTIMATED_SIZE[bb->object_type*2 + 1];

    float distance0 = (0.5*object_estimated_width)/(tan(object_width_angle*0.5));
    float distance1 = (0.5*object_estimated_height)/(tan(object_height_angle*0.5));

    return (distance0 + distance1)/2;
}


float axis_overlap(float a0, float a1, float b0, float b1) {
    if (a0 >= b0 && a1 <= b1) {
        return a1 - a0;
    }
    if (b0 >= a0 && b1 <= a1) {
        return b1 - b0;
    }
    if (a1 <= b0 || a0 >= b1) {
        return 0.0;
    } else if (a0 >= b0) {
        return b1 - a0;
    } else if (b0 >= a0) {
        return a1 - b0;
    } 
    return 0.0;
}


float intersection_over_union(bbox *bb0, bbox *bb1)
{
    float overlap_area = axis_overlap(bb0->x, bb0->x + bb0->w, bb1->x, bb1->x + bb1->w)*axis_overlap(bb0->y, bb0->y + bb0->h, bb1->y, bb1->y + bb1->h);
    float union_area = bb0->w*bb0->h + bb1->w*bb1->h - overlap_area;

    return overlap_area / union_area;
}

int suppress_bboxes(bbox *bboxes, int num_of_boxes, int index, float iou_treshold)
{
    for (int i = index+1; i < num_of_boxes; i++) {
        float iou = intersection_over_union(&bboxes[index], &bboxes[i]);
        if (iou > iou_treshold) {
            std::swap(bboxes[i], bboxes[num_of_boxes-1]);
            i--;
            num_of_boxes--;
        }
    }
    return num_of_boxes;
}

int sort_compare(const void *a_ptr, const void *b_ptr)
{
    bbox *a = (bbox*)a_ptr;
    bbox *b = (bbox*)b_ptr;

    if (a->confidence == b->confidence) {
        return 0;
    } else if (a->confidence > b->confidence) {
        return -1;
    } else {
        return 1;
    }
}

int non_maximum_suppression(bbox* bboxes, int num_of_boxes, float iou_treshold)
{
    for (int i = 0; i < num_of_boxes; i++) {
        qsort(bboxes, num_of_boxes, sizeof(bbox), &sort_compare);

        num_of_boxes = suppress_bboxes(bboxes, num_of_boxes, i, iou_treshold);
    }
    qsort(bboxes, num_of_boxes, sizeof(bbox), &sort_compare);
    return num_of_boxes;
}



tflite::MicroInterpreter *interpreter = nullptr;

extern "C" void init_yolo(void)
{
    static uint8_t *weights = (uint8_t *) heap_caps_malloc(sizeof(yolo_model_data), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    printf("Initializing yolo\n");
    printf("Weights: %dkB\n", sizeof(yolo_model_data) / 1024);

    if (weights == nullptr) {
        printf("Unable to allocate space for weights in PSRAM!\n");
        weights = (uint8_t*)yolo_model_data;
    } else {
        memcpy(weights, yolo_model_data, sizeof(yolo_model_data));
    }


    constexpr size_t tensor_arena_size = 1024*1024;
    static uint8_t *tensor_arena = (uint8_t *) heap_caps_malloc(tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    const tflite::Model* model = tflite::GetModel(weights);

    #ifdef USE_YOLOV5N6_XIAO

    static tflite::MicroMutableOpResolver<13> micro_op_resolver;
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddConcatenation();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddLogistic();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddPad();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddResizeNearestNeighbor();
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddSub();
    micro_op_resolver.AddTranspose();

    #endif

    #ifdef USE_YOLOV8N

    static tflite::MicroMutableOpResolver<14> micro_op_resolver;
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddConcatenation();
    micro_op_resolver.AddConv2D();
    //micro_op_resolver.AddDelegate();
    micro_op_resolver.AddLogistic();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddPad();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddResizeNearestNeighbor();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddSub();
    micro_op_resolver.AddTranspose();

    #endif


    printf("Creating MicroInterpreter\n");

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, tensor_arena_size);

    interpreter = &static_interpreter;

    printf("Allocating tensors\n");

    if (static_interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Failed to allocate tensors. Increase tensor arena size.\n");
        return;
    }


    TfLiteTensor *input = interpreter->input(0);
    TfLiteTensor *output = interpreter->output(0);
	
	printf("\nInput tensor: ");
    for (int i = 0; i < 4; i++) {
        printf("%d ", input->dims->data[i]);
    }
    printf("\nOutput tensor: ");
    for (int i = 0; i < 3; i++) {
        printf("%d ", output->dims->data[i]);
    }
    printf("\n");
}


void scale_input_bilinear(TfLiteTensor *input, uint8_t  *fb, int16_t w, int16_t h)
{
    int16_t imgz = input->dims->data[1];

    int8_t *out = input->data.int8;

    for (int16_t y = 0; y < imgz; y++) {
        for (int16_t x = 0; x < imgz; x++) {
            float fx = (float)x / (imgz-1);
            float fy = (float)y / (imgz-1);

            float ox = fx * (w-1);
            float oy = fy * (h-1);

            int16_t ix0 = (int16_t)ox;
            int16_t iy0 = (int16_t)oy;

            int16_t ix1 = std::min(ix0 + 1, w-1);
            int16_t iy1 = std::min(iy0 + 1, h-1);

            float wx1 = ox - ix0;
            float wy1 = oy - iy0; 

            float wx0 = 1.0f - wx1;
            float wy0 = 1.0f - wy1;

            for (int i = 0; i < 3; i++) {
                float c00 = fb[(iy0 * w + ix0)*3 + i];
                float c10 = fb[(iy0 * w + ix1)*3 + i];
                float c01 = fb[(iy1 * w + ix0)*3 + i];
                float c11 = fb[(iy1 * w + ix1)*3 + i];

                float c0 = (wx0 * c00) + (wx1 * c10);
                float c1 = (wx0 * c01) + (wx1 * c11);

                float c = (wy0 * c0) + (wy1 * c1);

                out[(y * imgz + x)*3 + i] = std::clamp((int)c - 128, -128, 127);
            }
        }
    }
}


void scale_input_nearest(TfLiteTensor *input, uint8_t  *fb, int32_t w, int32_t h)
{
    int32_t imgz = input->dims->data[1];

    int8_t *out = input->data.int8;

    for (int32_t y = 0; y < imgz; y++) {
        for (int32_t x = 0; x < imgz; x++) {
            int32_t nx = (x * w) / imgz;
            int32_t ny = (y * h) / imgz;

            int32_t r = fb[(ny * w + nx)*3 + 0];
            int32_t g = fb[(ny * w + nx)*3 + 1];
            int32_t b = fb[(ny * w + nx)*3 + 2];

            
            out[(y * imgz + x)*3 + 0] = r - 128;
            out[(y * imgz + x)*3 + 1] = g - 128;
            out[(y * imgz + x)*3 + 2] = b - 128;
        }
    }
}

inline float dequantize(int16_t value, TfLiteTensor *tensor)
{
    float f = (value - tensor->params.zero_point);
    return f*tensor->params.scale;
}


extern "C" int run_detector(uint8_t*fb, int32_t w, int32_t h, bbox *bboxes, int max_bboxes)
{
    TfLiteTensor *input = interpreter->input(0);

    uint64_t start = esp_timer_get_time();

    if (USE_BILINEAR_INTERP) {
        scale_input_bilinear(input, fb, w, h);
    } else {
        scale_input_nearest(input, fb, w, h);
    }


    uint64_t end = esp_timer_get_time();

    //printf("Scaling time: %ld ms\n", (int32_t)((end - start)/1000));

    start = esp_timer_get_time();

    if (kTfLiteOk != interpreter->Invoke()) {
        printf("Invoke failed.\n");
    }

    end = esp_timer_get_time();
    //printf("Inference time: %ld ms\n", (int32_t)((end - start)/1000));

    static uint64_t inference_total = 0;
    static uint64_t num_of_inferences = 0;

    inference_total += (end - start)/1000;
    num_of_inferences++;

    printf("Avg inference time: %ld ms\n", (int32_t)(inference_total / num_of_inferences));

    TfLiteTensor *output_tensor = interpreter->output(0);

    int8_t *output = output_tensor->data.int8;

    int detected_bboxes = 0;

    #ifdef USE_YOLOV5N6_XIAO

    int num_of_classes = output_tensor->dims->data[2] - 5;

    for (int32_t i = 0; i < output_tensor->dims->data[1]; i++) {
        int32_t index = i*output_tensor->dims->data[2];

        int8_t qx = output[index+0];
        int8_t qy = output[index+1];
        int8_t qw = output[index+2];
        int8_t qh = output[index+3];
        int8_t qc = output[index+4];

        float coinfidence = dequantize(qc, output_tensor);
        if (coinfidence > CONFIDENCE_TRESHOLD && detected_bboxes < MAX_TEMP_BBOX) {
            bbox *bb = &temp_bboxes[detected_bboxes];

            bb->confidence = coinfidence;

            bb->w = dequantize(qw, output_tensor);
            bb->h = dequantize(qh, output_tensor);

            bb->x = dequantize(qx, output_tensor) - bb->w*0.5f;
            bb->y = dequantize(qy, output_tensor) - bb->h*0.5f;

            bb->object_type = 0;
            int8_t max_prob = output[index+5];
            for (int c = 0; c < num_of_classes; c++) {
                if (output[index+5+c] > max_prob) {
                    bb->object_type = c;
                    max_prob = output[index+5+c];
                }
            }

            detected_bboxes++;
        }
    }
    #endif

    #ifdef USE_YOLOV8N

    int num_of_classes = output_tensor->dims->data[1] - 4;

    for (int32_t i = 0; i < output_tensor->dims->data[2]; i++) {
        int8_t vec[128];
        for (int j = 0; j < 4+num_of_classes; j++) {
            vec[j] = output[j*output_tensor->dims->data[2]+i];
        }

        int8_t qx = vec[0];
        int8_t qy = vec[1];
        int8_t qw = vec[2];
        int8_t qh = vec[3];
 
        int cp = vec[4];
        int best_class = 0;

        for (int j = 0; j < num_of_classes; j++) {
            if (vec[j+4] > cp) {
                best_class = j;
                cp = vec[j+4];
            }
        }

        float probability = dequantize(cp, output_tensor);
        if (probability > CONFIDENCE_TRESHOLD && detected_bboxes < MAX_TEMP_BBOX) {
            bbox *bb = &temp_bboxes[detected_bboxes];

            bb->confidence = probability;

            bb->w = dequantize(qw, output_tensor);
            bb->h = dequantize(qh, output_tensor);

            bb->x = dequantize(qx, output_tensor) - bb->w*0.5f;
            bb->y = dequantize(qy, output_tensor) - bb->h*0.5f;

            bb->object_type = best_class;

            detected_bboxes++;
        }
    }
    #endif
    
    int num_of_bboxes = non_maximum_suppression(temp_bboxes, detected_bboxes, IOU_TRESHOLD);

    if (num_of_bboxes > max_bboxes) {
        num_of_bboxes = max_bboxes;
    }

    for (int i = 0; i < num_of_bboxes; i++) {
        bboxes[i] = temp_bboxes[i];
        bboxes[i].estimated_distance = estimate_distance(&bboxes[i]);
    }

    return num_of_bboxes;
}