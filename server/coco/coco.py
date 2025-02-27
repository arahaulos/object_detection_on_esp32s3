import json

class coco_label:
    def __init__(self, image_id, category_id, category_name, bbox):
        self.image_id = image_id
        self.category_id = category_id
        self.category_name = category_name
        self.bbox = bbox

class coco_category:
    def __init__(self, category_id, name):
        self.category_id = category_id
        self.name = name

class coco_image:
    def __init__(self, image_id, filepath, width, height):
        self.image_id = image_id
        self.filepath = filepath
        self.width = width
        self.height = height

class coco_dataset:
    def __init__(self, annonations_file, files_folder):
        self.labels = {}
        self.categories = {}
        self.images = {}

        with open(annonations_file, 'r') as file:
            data = json.load(file)

            for item in data["categories"]:
                category_id = item['id']
                self.categories[category_id] = coco_category(category_id, item['name'])

            for item in data["images"]:
                image_id = item['id']
                self.images[image_id] = coco_image(image_id, files_folder + item['file_name'], item['width'], item['height'])
                self.labels[image_id] = []


            for item in data['annotations']:
                image_id = item['image_id']
                self.labels[image_id].append(coco_label(image_id, item['category_id'], self.categories[item['category_id']].name, item['bbox']))

    def get_images_containing(self, label_names):
        list = []

        for image_id in self.images:
            contains = False

            for label in self.labels[image_id]:
                if (label.category_name in label_names):
                    contains = True
                    break

            if (contains):
                list.append(image_id)

        return list
    
    def get_images_not_containing(self, label_names):
        list = []

        for image_id in self.images:
            contains = False

            for label in self.labels[image_id]:
                if (label.category_name not in label_names):
                    contains = True
                    break

            if (contains):
                list.append(image_id)

        return list
    
    def get_image_ids(self):
        list = []

        for image_id in self.images:
            list.append(image_id)

        return list
    
    def filter_labels(self, label_names):
        for image_id in self.images:
            newlist = []
            for label in self.labels[image_id]:
                if (label.category_name in label_names):
                    newlist.append(label)

            self.labels[image_id] = newlist

