import numpy as np
import tensorflow as tf

from PIL import Image

import bbox
import nms
import distance_estimation


def dequantize(value, quant):
    (mult, sub) = quant

    newvalue = float(value)
    return (newvalue - sub)*mult

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

        self.output_quant = (output_details[0]["quantization"][0], output_details[0]["quantization"][1])

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

        input_shape = input_details[0]['shape']
        output_shape = output_details[0]['shape']

        num_of_classes = output_shape[1] - 4

        img = np.array(input_image.resize((input_shape[1], input_shape[2])))

        input_data = np.array(np.zeros(input_shape), dtype=np.int8)

        input_data[0][:] = img - 128

        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        for i in range(output_shape[2]):
            rx = dequantize(output_data[0][0][i], self.output_quant)
            ry = dequantize(output_data[0][1][i], self.output_quant)

            rw = dequantize(output_data[0][2][i], self.output_quant)
            rh = dequantize(output_data[0][3][i], self.output_quant)

            for c in range(num_of_classes):
                cp = dequantize(output_data[0][4+c][i], self.output_quant)

                if (cp > confidence_treshold):
                    bboxes.append(bbox.bbox(0, cp, 0.0, rx - rw/2, ry - rh/2, rw, rh))


        bboxes = nms.non_maximum_suppression(bboxes)

        for bb in bboxes:
            bb.estimated_distance = distance_estimation.estimate_distance(bb, 0.6, 1.75)

        return bboxes