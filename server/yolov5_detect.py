import numpy as np
import tensorflow as tf

from PIL import Image
from PIL import ImageDraw

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

        self.output_quant = (output_details[0]["quantization"][0], output_details[0]["quantization"][1])

        operations = get_tflite_operations(self.interpreter)

        print("Operations used in the model:")
        for op in sorted(operations):
            print(op)

    def detect(self, input_image, confidence_treshold = 0.25):
        bboxes = []

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        output_shape = output_details[0]['shape']

        img = np.array(input_image.resize((input_shape[1], input_shape[2]), Image.Resampling.NEAREST))

        input_data = np.array(np.zeros(input_shape), dtype=np.int8)

        input_data[0][:] = img - 128

        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        for i in range(output_shape[1]):
            tensor = output_data[0][i]

            x, y, w, h, cp = tensor[:5]
            c = tensor[5:]

            cpf = dequantize(cp, self.output_quant)
            xf = dequantize(x, self.output_quant)
            yf = dequantize(y, self.output_quant)
            wf = dequantize(w, self.output_quant)
            hf = dequantize(h, self.output_quant)

            if (cpf > confidence_treshold):
                best_class_val = c[0]
                best_class = 0
                for k in range(len(c)):
                    if (c[k] > best_class_val):
                        best_class_val = c[k]
                        best_class = k

                if (cpf > confidence_treshold):
                    bboxes.append(bbox.bbox(best_class, cpf, 0.0, xf - wf/2, yf - hf/2, wf, hf))

            
        bboxes = nms.non_maximum_suppression(bboxes)

        for bb in bboxes:
            bb.estimated_distance = distance_estimation.estimate_distance(bb, 0.6, 1.75)

        return bboxes
        
