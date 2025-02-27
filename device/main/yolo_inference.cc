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

#include "yolov5.h"

#include <esp_heap_caps.h>

constexpr float CONFIDENCE_TRESHOLD = 0.25f;
constexpr float IOU_TRESHOLD = 0.25f;
constexpr int NUM_OF_CLASSES = 1;

constexpr int MAX_BBOX = 64;
detected_bbox bboxes[MAX_BBOX];


constexpr float OBJECT_ESTIMATED_SIZE[NUM_OF_CLASSES*2] =
{
    0.6f, 1.75f //Person
};


float estimate_distance(detected_bbox *bbox, float vertical_fov = 66.0f*M_PI/180.0f, float aspect_ratio = 320.0f/240.0f)
{
    float horizontal_fov = vertical_fov / aspect_ratio;

    float object_width_angle = bbox->w * vertical_fov;
    float object_height_angle = bbox->h * horizontal_fov;

    float object_estimated_width = OBJECT_ESTIMATED_SIZE[bbox->object_type*2 + 0];
    float object_estimated_height = OBJECT_ESTIMATED_SIZE[bbox->object_type*2 + 1];

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


float intersection_over_union(detected_bbox *bbox0, detected_bbox *bbox1)
{
    float overlap_area = axis_overlap(bbox0->x, bbox0->x + bbox0->w, bbox1->x, bbox1->x + bbox1->w)*axis_overlap(bbox0->y, bbox0->y + bbox0->h, bbox1->y, bbox1->y + bbox1->h);
    float union_area = bbox0->w*bbox0->h + bbox1->w*bbox1->h - overlap_area;

    return overlap_area / union_area;
}

int suppress_bboxes(detected_bbox *bboxes, int num_of_boxes, int index, float iou_treshold)
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
    detected_bbox *a = (detected_bbox*)a_ptr;
    detected_bbox *b = (detected_bbox*)b_ptr;

    if (a->confidence == b->confidence) {
        return 0;
    } else if (a->confidence > b->confidence) {
        return -1;
    } else {
        return 1;
    }
}

int non_maximum_suppression(detected_bbox* bboxes, int num_of_boxes, float iou_treshold)
{
    for (int i = 0; i < num_of_boxes; i++) {
        qsort(bboxes, num_of_boxes, sizeof(detected_bbox), &sort_compare);

        num_of_boxes = suppress_bboxes(bboxes, num_of_boxes, i, iou_treshold);
    }
    return num_of_boxes;
}



tflite::MicroInterpreter *interpreter = nullptr;

extern "C" void init_yolo(void)
{
    static uint8_t *weights = (uint8_t *) heap_caps_malloc(sizeof(yolov5_weights), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    printf("Initializing yolo\n");
    printf("Weights: %dkB\n", sizeof(yolov5_weights) / 1024);

    if (weights == nullptr) {
        printf("Unable to allocate space for weights in PSRAM!\n");
        weights = (uint8_t*)yolov5_weights;
    } else {
        memcpy(weights, yolov5_weights, sizeof(yolov5_weights));
    }


    constexpr size_t tensor_arena_size = 512*1024;
    static uint8_t *tensor_arena = (uint8_t *) heap_caps_malloc(tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    const tflite::Model* model = tflite::GetModel(weights);

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
    

    printf("test");
}

inline float dequantize(int16_t value, TfLiteTensor *tensor)
{
    float f = (value - tensor->params.zero_point);
    return f*tensor->params.scale;
}

extern "C" detected_bbox* run_detector(uint8_t*fb, int32_t w, int32_t h, uint32_t *num_of_bboxes)
{
    TfLiteTensor *input = interpreter->input(0);
    int8_t *input_ptr = input->data.int8;

    for (int32_t y = 0; y < 192; y++) {
        for (int32_t x = 0; x < 192; x++) {
            int32_t nx = (x * w) / 192;
            int32_t ny = (y * h) / 192;

            int32_t r = fb[(ny * w + nx)*3 + 0];
            int32_t g = fb[(ny * w + nx)*3 + 1];
            int32_t b = fb[(ny * w + nx)*3 + 2];

            
            input_ptr[(y * 192 + x)*3 + 0] = r - 128;
            input_ptr[(y * 192 + x)*3 + 1] = g - 128;
            input_ptr[(y * 192 + x)*3 + 2] = b - 128;
        }
    }

    uint64_t start = esp_timer_get_time();

    if (kTfLiteOk != interpreter->Invoke()) {
        printf("Invoke failed.\n");
    }

    uint64_t end = esp_timer_get_time();
    printf("Inference time: %ld ms\n", (int32_t)((end - start)/1000));


    TfLiteTensor *output_tensor = interpreter->output(0);

    //printf("Bytes: %d,  Scale: %f  Zero: %ld\n", output_tensor->bytes, output_tensor->params.scale, output_tensor->params.zero_point);

    int8_t *output = output_tensor->data.int8;

    int detected_bboxes = 0;

    static const int32_t output_heads[4] = {24, 12, 6, 3};

    int32_t head_index = 0;

    for (int32_t i = 0; i < 4; i++) {
        int32_t grid_res = output_heads[i];

        for (int32_t j = 0; j < 3; j++) {
            for (int32_t k = 0; k < grid_res*grid_res; k++) {
                int32_t index = (head_index + k)*(5+NUM_OF_CLASSES);

                int8_t qx = output[index+0];
                int8_t qy = output[index+1];
                int8_t qw = output[index+2];
                int8_t qh = output[index+3];
                int8_t qc = output[index+4];

                float coinfidence = dequantize(qc, output_tensor);
                if (coinfidence > CONFIDENCE_TRESHOLD && detected_bboxes < MAX_BBOX) {
                    detected_bbox *bbox = &bboxes[detected_bboxes];

                    bbox->confidence = coinfidence;

                    bbox->w = dequantize(qw, output_tensor);
                    bbox->h = dequantize(qh, output_tensor);

                    bbox->x = dequantize(qx, output_tensor) - bbox->w*0.5f;
                    bbox->y = dequantize(qy, output_tensor) - bbox->h*0.5f;

                    bbox->object_type = 0;
                    int8_t max_prob = output[index+5];
                    for (int c = 0; c < NUM_OF_CLASSES; c++) {
                        if (output[index+5+c] > max_prob) {
                            bbox->object_type = c;
                            max_prob = output[index+5+c];
                        }
                    }

                    detected_bboxes++;
                }
            }
            head_index += grid_res*grid_res;
        }
    }
    
    *num_of_bboxes = non_maximum_suppression(bboxes, detected_bboxes, IOU_TRESHOLD);

    for (int i = 0; i < *num_of_bboxes; i++) {
        bboxes[i].estimated_distance = estimate_distance(&bboxes[i]);

        printf("%f %f\n", bboxes[i].x, bboxes[i].y);
    }

    return bboxes;
}