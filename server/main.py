from PIL import Image
from PIL import ImageDraw

import matplotlib.pyplot as plt
import pygame

import yolov5_detect
import yolov8_detect
import time
import server


def draw_bboxes(image, bboxes, color):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for bbox in bboxes:
        nbb = bbox.scaled(width, height)

        edist = bbox.estimated_distance

        draw.rectangle([(nbb.x, nbb.y), (nbb.x + nbb.w, nbb.y + nbb.h)], outline=color, width=8)

        draw.text((nbb.x, nbb.y), "{0:.3g}".format(edist))
    
    return image


def main_loop():

    #yolo = yolov5_detect.yolov5_detect("last-int8.tflite")
    yolo = yolov8_detect.yolov8_detect("yolov8n_full_integer_quant.tflite", output_grids=[40, 20, 10])
    serv = server.server("192.168.1.101", 6969, yolo)

    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))

    running = True
    while running:
        #image = Image.open("test.jpg")
        #start = time.time()
        #bboxes = yolo.detect(image)
        #inference_time = time.time() - start
        #print("{}ms".format(inference_time*1000))

        #draw_bboxes(image, bboxes, "red")
        #surf = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
        #screen.blit(surf, (0, 0))


        image = serv.get_last_received_image()
        bboxes = serv.get_last_received_bboxes()

        if (image != None):
            image = image.resize((width, height))
            if (bboxes != None):
                draw_bboxes(image, bboxes, "red")

            width, height = image.size

            surf = pygame.image.fromstring(image.tobytes(), image.size, image.mode)

            screen.blit(surf, (0, 0))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        time.sleep(0.0166)
    
    serv.close()
    pygame.quit()

if __name__ == "__main__":
    main_loop()