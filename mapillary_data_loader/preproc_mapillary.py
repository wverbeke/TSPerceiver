import os
import sys
main_directory = os.path.dirname(os.path.dirname(os.path.abspath( __file__)))
sys.path.insert(0, main_directory)

from tqdm import tqdm

import json
import math
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from mapillary_data_loader.bbox import Bbox
import torch

# Data source of full images and annotations.
SOURCE_PATH = "/home/willem/code/bayesian-evaluation/datasets/mapillary/"
SOURCE_ANNOTATION_DIRECTORY = os.path.join(SOURCE_PATH, "mtsd_v2_fully_annotated/annotations/")
SOURCE_TRAIN_IMAGES_DIRECTORY = os.path.join(SOURCE_PATH, "images_train/")
SOURCE_EVAL_IMAGES_DIRECTORY = os.path.join(SOURCE_PATH, "images_val/")
SOURCE_TEST_IMAGES_DIRECTORY = os.path.join(SOURCE_PATH, "images_test/")

# Output directories, relative to the working directory.
TRAIN_PATCH_DIRECTORY = os.path.join(main_directory, "train_patches/")
EVAL_PATCH_DIRECTORY = os.path.join(main_directory, "eval_patches/")
TEST_PATCH_DIRECTORY = os.path.join(main_directory, "test_patches/")
TRAIN_ANNOTATION_LIST_PATH = os.path.join(main_directory, "annotations/train_annotations.json")
EVAL_ANNOTATION_LIST_PATH = os.path.join(main_directory, "annotations/eval_annotations.json")
TEST_ANNOTATION_LIST_PATH = os.path.join(main_directory, "annotations/test_patches/train_annotations.json")

# Hyperparameters of the preprocessing.
PADDING_FRACTION=0.2


def read_annotation(annotation_path):
    with open(annotation_path, "r") as f:
        return json.load(f)


class ImageDataSet(Dataset):

    def __init__(self, root: str):
        self._root = root
        self._images = os.listdir(self._root)
        self._transform = transforms.ToTensor()

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):

        # Read image.
        image_path = os.path.join(self._root, self._images[index])
        image = Image.open(image_path).convert("RGB")
        image = self._transform(image)

        # Read annotation.
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = os.path.join(SOURCE_ANNOTATION_DIRECTORY, image_name + ".json")
        annotation = read_annotation(annotation_path)
        return image, annotation, image_name


def extract_patches_and_annotation(image, annotation):
    signs = annotation["objects"]
    patches = []
    class_names = []
    for s in signs:
        bbox_dict = s["bbox"]

        # Annotations with this property cross the boundary of a 360 camera.
        # The cropping logic used thusfar will not work for such boxes.
        if "cross_boundary" in s["bbox"]:
            continue

        bbox = Bbox.from_dict(s["bbox"])
        p = bbox.crop_from_image(image)
        patches.append(bbox.crop_from_image(image, padding_fraction=PADDING_FRACTION))

        class_name = s["label"]
        class_names.append(class_name)
    
    return patches, class_names


def preprocess_images(input_directory: str, output_directory: str, output_annotation_path: str):

    # Make the output directory if necessary.
    os.makedirs(output_directory, exist_ok=True)
    anno_dir = os.path.dirname(output_annotation_path)
    os.makedirs(anno_dir, exist_ok=True)

    # Make the data loader returning full-sized Mapillary images and their annotations in a dictionary.
    dataset = ImageDataSet(root=input_directory)
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=os.cpu_count(), prefetch_factor=5)

    # Store a mapping of all cropped traffic sign patch names to the corresponding class name.
    patch_name_to_class_name = {}

    for image, annotation, image_name in tqdm(data_loader):
        # For some reason the unpacking returns (image_name,) probably because of pytorch's collate
        image_name = image_name[0]
        patches, class_names = extract_patches_and_annotation(image, annotation)
        for i, (patch, class_name) in enumerate(zip(patches, class_names)):
            patch_path = os.path.join(output_directory, f"{image_name}_patch_{i}.png")
            patch_name_to_class_name[patch_path] = class_name
            save_image(patch, patch_path)

    with open(output_annotation_path, "w") as f:
        json.dump(patch_name_to_class_name, f)


def preprocess_train_images():
    return preprocess_images(input_directory=SOURCE_TRAIN_IMAGES_DIRECTORY, output_directory=TRAIN_PATCH_DIRECTORY, output_annotation_path=TRAIN_ANNOTATION_LIST_PATH)


def preprocess_eval_images():
    return preprocess_images(input_directory=SOURCE_EVAL_IMAGES_DIRECTORY, output_directory=EVAL_PATCH_DIRECTORY, output_annotation_path=EVAL_ANNOTATION_LIST_PATH)


def preprocess_test_images():
    return preprocess_images(input_directory=SOURCE_TEST_IMAGES_DIRECTORY, output_directory=TEST_PATCH_DIRECTORY, output_annotation_path=TEST_ANNOTATION_LIST_PATH)


if __name__ == "__main__":
    preprocess_train_images()
    preprocess_eval_images()
    #preprocess_test_images()
