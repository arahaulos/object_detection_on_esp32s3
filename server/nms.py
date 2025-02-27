import bbox

def suppress_bboxes(bboxes, index, iou_treshold):
    bb = bboxes[index]
    boxes_left = len(bboxes)
    index += 1
    while index < boxes_left:
        if (bb.intersection_of_union(bboxes[index]) > iou_treshold):
            bboxes.remove(bboxes[index])
            index -= 1

        boxes_left = len(bboxes)
        index += 1
        
        


def non_maximum_suppression(bboxes, iou_treshold = 0.25):
    index = 0
    while True:
        boxes_left = len(bboxes)
        if (index >= boxes_left-1):
            break

        bboxes = sorted(bboxes, key=lambda x: x.confidence, reverse=True)

        suppress_bboxes(bboxes, index, iou_treshold)
        
        index += 1

    return bboxes
