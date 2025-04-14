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

        draw.rectangle([(nbb.x, nbb.y), (nbb.x + nbb.w, nbb.y + nbb.h)], outline=color, width=4)

        draw.text((nbb.x, nbb.y), "{0:.3g}".format(bbox.estimated_distance))
    
    return image


def main_loop():

    #yolo = yolov5_detect.yolov5_detect("last-int8.tflite")
    yolo = yolov8_detect.yolov8_detect("yolov8n_full_integer_quant.tflite")
    serv = server.server("192.168.1.101", 6969, yolo)

    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))

    running = True
    while running:
        image = serv.get_last_received_image()
        bboxes = serv.get_last_received_bboxes()

        if (image != None):
            image = image.resize((width, height))
            #bboxes2 = yolo.detect(image)

            if (bboxes != None):
                image = draw_bboxes(image, bboxes, "red")

            #if (bboxes2 != None):
            #    image = draw_bboxes(image, bboxes2, "green")

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