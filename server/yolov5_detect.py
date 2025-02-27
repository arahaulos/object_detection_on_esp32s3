import numpy as np
import tensorflow as tf

from PIL import Image
from PIL import ImageDraw

import bbox
import nms
import distance_estimation

def dequantize(value):
    newvalue = float(value)
    return (newvalue + 127)*0.01092343870550394

def get_tflite_operations(interpreter):
    op_details = interpreter._get_ops_details() 
    op_types = {op["op_name"] for op in op_details}

    return op_types

class yolov5_detect:
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

    def detect(self, input_image, confidence_treshold = 0.25):
        bboxes = []

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        img = np.array(input_image.resize((192, 192), Image.Resampling.NEAREST))

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

        head_dims = [24, 12, 6, 3]

        index = 0
        for dl in range(4):

            grid_dim = head_dims[dl]

            for i in range(grid_dim*grid_dim):

                for j in range(3):
                    tensor = output_data[0][index + j*grid_dim*grid_dim + i]

                    x, y, w, h, cp = tensor[:5]
                    c = tensor[5:]

                    cpf = dequantize(cp)
                    xf = dequantize(x)
                    yf = dequantize(y)
                    wf = dequantize(w)
                    hf = dequantize(h)

                    if (cpf > confidence_treshold):
                        best_class_val = c[0]
                        best_class = 0
                        for k in range(len(c)):
                            if (c[k] > best_class_val):
                                best_class_val = c[k]
                                best_class = k

                        bboxes.append(bbox.bbox(best_class, cpf, 0.0, xf - wf/2, yf - hf/2, wf, hf))

            index += grid_dim*grid_dim*3

            bboxes = nms.non_maximum_suppression(bboxes)

            for bbox in bboxes:
                bbox.estimated_distance = distance_estimation.estimate_distance(bbox, 0.6, 1.75)

            return bboxes
        
