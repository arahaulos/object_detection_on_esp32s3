import coco
import shutil
import os

def convert_coco_labels(coco_files_path, coco_annonations_file, images_output_folder, labels_output_folder, images_file, labels_to_extract):

    print("Loading coco dataset {}".format(coco_annonations_file))

    dataset = coco.coco_dataset(coco_annonations_file, coco_files_path)

    new_category_id_map = {}

    for category_id in dataset.categories:
        category = dataset.categories[category_id]
        if (category.name in labels_to_extract):
            new_category_id_map[category.category_id] = labels_to_extract.index(category.name)
    

    print("Filtering labels");
    dataset.filter_labels(labels_to_extract)

    image_ids = dataset.get_images_containing(labels_to_extract)


    common_prefix = os.path.commonpath([images_output_folder, images_file])
    relative_path = os.path.relpath(images_output_folder, common_prefix).replace("\\", "/")

    with open(images_file, "w") as images_file:
        new_id = 0
        for image_id in image_ids:
            images_file.write("./{}/{}.jpg\n".format(relative_path, new_id))

            old_file_path = dataset.images[image_id].filepath
            new_file_name = "{}{}.jpg".format(images_output_folder, new_id)
            new_label_file_name = "{}{}.txt".format(labels_output_folder, new_id)

            shutil.copy(old_file_path, new_file_name)

            labels = dataset.labels[image_id]
            image = dataset.images[image_id]

            with open(new_label_file_name, "w") as label_file:
                for label in labels:
                    label_file.write("{} {} {} {} {}\n".format(new_category_id_map[label.category_id], 
                                                               (label.bbox[0] + label.bbox[2]/2) / image.width, 
                                                               (label.bbox[1] + label.bbox[3]/2) / image.height, 
                                                               label.bbox[2] / image.width, 
                                                               label.bbox[3] / image.height))

            new_id += 1
            if (new_id % 100 == 0):
                print("\rCopying file {}%    ".format(int(new_id*100 / len(image_ids))), end='')

    print("\nNew dataset created!")


convert_coco_labels("data/train2017/", "data/annotations/instances_train2017.json", "newdataset2/images/train2017/", "newdataset2/labels/train2017/", "newdataset2/train2017.txt", ["person", "bicycle", "car"])
convert_coco_labels("data/val2017/", "data/annotations/instances_val2017.json", "newdataset2/images/val2017/", "newdataset2/labels/val2017/", "newdataset2/val2017.txt", ["person", "bicycle", "car"])