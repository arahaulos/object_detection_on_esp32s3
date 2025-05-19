import coco.coco
import bbox
from PIL import Image
import yolov5_detect
import yolov8_detect
import numpy as np


def calculate_average_precision(model, dataset, object_class, iou_treshold = 0.5):
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
            if (pbbox.object_type == object_class):
                pbbox.image_id = image_id
                all_predictions.append(pbbox)

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

    return ap






        



def main():
    file1 = open("ap.txt", "w")

    coco_validation = coco.coco.coco_dataset("coco/data/annotations/instances_val2017.json", "coco/data/val2017/")
    coco_validation.filter_labels(["person"])
    coco_validation.filter_images(["person"])

    yolov5_models = ["yolov5n6-xiao-192x192.tflite",
                     "yolov5n6-xiao-256x256.tflite"]
    
    yolov8_models = ["yolov8n_192x192.tflite",
                     "yolov8n_256x256.tflite"]

    for m in yolov5_models:
        yolo = yolov5_detect.yolov5_detect(m)
        ap = calculate_average_precision(yolo, coco_validation, 0)
    
        file1.write("{} {}\n".format(m, ap))


    for m in yolov8_models:
        yolo = yolov8_detect.yolov8_detect(m)
        ap = calculate_average_precision(yolo, coco_validation, 0)

        file1.write("{} {}\n".format(m, ap))

    file1.close()


if __name__ == "__main__":
    main()

