"""Data loader for Mapillary.

The loader will load images containing individual traffic sign patches. This assumes that the Mapillary
data set has already been preprocessed to extract all traffic signs.
"""
import os
import json
import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision

from torchvision import datasets, transforms

from mapillary_data_loader.load_mapillary import MapillaryDataset
from mapillary_data_loader.make_class_list import mapillary_class_list
from mapillary_data_loader.preproc_mapillary import TRAIN_ANNOTATION_LIST_PATH, EVAL_ANNOTATION_LIST_PATH, read_annotation

_SHARED_TRANSFORMS = transforms.ToTensor()
_DATALOADER_KWARGS = {"num_workers": os.cpu_count(), "prefetch_factor": 4}
_TRANSFORMS = transforms.Compose([transforms.RandomRotation(15), _SHARED_TRANSFORMS])#transforms.RandomCrop(TRAINING_PATCH_SIZE),  _SHARED_TRANSFORMS



class MapillaryDatasetBase(Dataset):
    """Shared operations between CNN and Perceiver dataloader."""

    # kwargs are ignored but make it callable in the same way as standard pytorch loaders
    def __init__(self, train: bool, **kwargs):
        if train:
            annotation_dict = read_annotation(TRAIN_ANNOTATION_LIST_PATH)
        else:
            annotation_dict = read_annotation(EVAL_ANNOTATION_LIST_PATH)
        self._image_paths = []
        self._annotations = []

        # The class list is made from the set of annotations if it does not already exist.
        class_list = mapillary_class_list()

        # All patch paths are stored with their respective annotation.
        for image_path, class_name in annotation_dict.items():

            # The annotation is stored as a list of size 1.
            # TODO: Fix this.
            class_name = class_name[0]
            self._image_paths.append(image_path)
            self._annotations.append(class_list.index(class_name))
        #self._transform = _TRANSFORMS

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index):

        # Read and return the image.
        image = Image.open(self._image_paths[index]).convert("RGB")
        return image, self._annotations[index]
        #image_tensor = self._transform(image)

        #return image_tensor, self._annotations[index]


class MapillaryDatasetPerceiver(MapillaryDatasetBase):

    def __init__(self, max_size=90000, **kwargs):
        super.__init__(**kwargs)
        self._max_size = max_size
        im_dim = int(math.sqrt(max_size))
        self._resize_op = torchvision.transforms.Resize((im_dim, im_dim))

    def _process_im(self, im):
        im_size = im.shape[1]*im.shape[2]
        if im_size <= max_size:
            im = im.view(im.shape[0], im.shape[1]*im.shape[2])
            if im__size == max_size:
                return im

            diff = max_size - im.shape[1]
            pad = torch.full([im.shape[0], im.shape[1]], -1)
            im = torch.cat([im, pad])
            return im
        else:
            im = self._resize_op(im)
            return self._process_im(im)


    def __getitem__(self, index):
        im, anno = super().__getitem__(index)
        return self._process_im(im), anno


class MapillaryDatasetCNN(self, im_size, **kwargs):
    def __init__(**kwargs):
        self._resize = torchvision.transforms.Resize(im_size)

    def __getitem__(self, im):
        im, anno = super().__getitem__(index)
        return self._resize(im), anno



if __name__ == "__main__":
    ds = MapillaryDatasetPerceiver(900, True)
    for d in ds:
        print(d)

