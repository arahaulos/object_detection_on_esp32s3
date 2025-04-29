import coco.coco
import bbox
from PIL import Image
import yolov5_detect
import numpy as np


def test_model_accuracy(model, dataset, iou_treshold = 0.5):
    images_tested = 0

    total_ground_truths = 0

    all_predictions = []
    all_ground_truths_for_images = {}

    for image_id in dataset.images:
        ground_truth_boxes = []
        image = dataset.images[image_id]
        for label in dataset.labels[image_id]:
            ground_truth_boxes.append(bbox.bbox(label.category_id, 
                                                1.0, 
                                                0.0, 
                                                label.bbox[0] / image.width,
                                                label.bbox[1] / image.height,
                                                label.bbox[2] / image.width,
                                                label.bbox[3] / image.height))
            
        all_ground_truths_for_images[image_id] = ground_truth_boxes
        total_ground_truths += len(ground_truth_boxes)

        loaded_image = Image.open(image.filepath).convert("RGB")

        predicted_boxes = model.detect(loaded_image, confidence_treshold=0.0)
        for pbbox in predicted_boxes:
            pbbox.image_id = image_id

        all_predictions.extend(predicted_boxes)

        images_tested += 1
        print("Predicting: {}/{}".format(images_tested, len(dataset.images)))

    all_predictions.sort(key=lambda x: x.confidence, reverse=True) 

    false_positive = 0
    true_positive = 0

    precision = []
    recall = []

    for pbbox in all_predictions:
        ground_truth_boxes = all_ground_truths_for_images[pbbox.image_id]

        best_iou = 0.0
        matched_truth_box = None
        for tbbox in ground_truth_boxes:
            iou = pbbox.intersection_of_union(tbbox)
            if (iou > best_iou):
                best_iou = iou
                matched_truth_box = tbbox

        if (best_iou > iou_treshold):
            true_positive += 1
            ground_truth_boxes.remove(matched_truth_box)
        else:
            false_positive += 1

        precision.append(true_positive / (true_positive + false_positive))
        recall.append(true_positive / total_ground_truths)

    precision = np.array(precision)
    recall = np.array(recall)

    recall_levels = np.linspace(0, 1, 101)
    ap = 0
    for r in recall_levels:
        p = np.max(precision[recall >= r]) if np.any(recall >= r) else 0
        ap += p
    ap /= 101

    print(ap)






        



def main():
    #def __init__(self, annonations_file, files_folder):
    #convert_coco_labels("data/val2017/", "data/annotations/instances_val2017.json", "newdataset2/images/val2017/", "newdataset2/labels/val2017/", "newdataset2/val2017.txt", ["person", "bicycle", "car"])

    coco_validation = coco.coco.coco_dataset("coco/data/annotations/instances_val2017.json", "coco/data/val2017/")
    coco_validation.filter_labels(["person"])
    coco_validation.filter_images(["person"])

    yolo = yolov5_detect.yolov5_detect("yolov5n6-xiao-256x256.tflite")

    test_model_accuracy(yolo, coco_validation)


if __name__ == "__main__":
    main()

