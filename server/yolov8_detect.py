import numpy as np
import tensorflow as tf

from PIL import Image

import bbox
import nms
import distance_estimation


def dequantize(value):
    newvalue = float(value)
    return (newvalue + 128.0)*0.00503881461918354



def get_tflite_operations(interpreter):
    op_details = interpreter._get_ops_details() 
    op_types = {op["op_name"] for op in op_details}

    return op_types

class yolov8_detect:
    def __init__(self, tflite_model_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        print(input_details)
        print(output_details)

        print("input quantization")
        print(input_details[0]["quantization"])

        print("output quantization")
        print(output_details[0]["quantization"])

        operations = get_tflite_operations(self.interpreter)

        print("Operations used in the model:")
        for op in sorted(operations):
            print(op)

    def detect(self, input_image, confidence_treshold = 0.5):
        bboxes = []

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        img = np.array(input_image.resize((192, 192)))

        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.int8)

        for y in range(192):
            for x in range(192):
                input_data[0][y][x][0] = img[y][x][0] - 128
                input_data[0][y][x][1] = img[y][x][1] - 128
                input_data[0][y][x][2] = img[y][x][2] - 128

        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        grid_sizes = [24, 12, 6]
        index = 0
        for grid_size in grid_sizes:

            for i in range(grid_size*grid_size):
                rx = dequantize(output_data[0][0][index+i])*192
                ry = dequantize(output_data[0][1][index+i])*192

                rw = dequantize(output_data[0][2][index+i])*192
                rh = dequantize(output_data[0][3][index+i])*192

                for c in range(80):
                    cp = dequantize(output_data[0][4+c][index+i])

                    if (cp > confidence_treshold):
                        bboxes.append(bbox.bbox(0, cp, 0.0, rx - rw/2, ry - rh/2, rx + rw/2, ry + rh/2))


            index += grid_size*grid_size

        bboxes = nms.non_maximum_suppression(bboxes)

        for bbox in bboxes:
            bbox.estimated_distance = distance_estimation.estimate_distance(bbox, 0.6, 1.75)

        return bboxes