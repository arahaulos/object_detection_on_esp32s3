
import bbox
import math

def estimate_distance(box, object_estimated_width, object_estimated_height, vertical_fov = 66*math.pi/180, aspect_ratio = 320.0/240.0):
    horizontal_fov = vertical_fov / aspect_ratio

    object_width_angle = box.w * vertical_fov
    object_height_angle = box.h * horizontal_fov

    distance0 = (0.5*object_estimated_width)/(math.tan(object_width_angle*0.5))
    distance1 = (0.5*object_estimated_height)/(math.tan(object_height_angle*0.5))

    return (distance0 + distance1)/2