"""Data loader for Mapillary.

The loader will load images containing individual traffic sign patches. This assumes that the Mapillary
data set has already been preprocessed to extract all traffic signs.
"""
import os
import sys
from typing import Tuple
main_directory = os.path.dirname(os.path.dirname(os.path.abspath( __file__)))
sys.path.insert(0, main_directory)

import json
import torch
import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import einops

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from random_fractional_crop import RandomFractionalCrop
from torch_list import TorchList

from mapillary_data_loader.make_class_list import mapillary_class_list
from mapillary_data_loader.preproc_mapillary import TRAIN_ANNOTATION_LIST_PATH, EVAL_ANNOTATION_LIST_PATH, read_annotation

_EVAL_TRANSFORMS = transforms.ToTensor()
_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomRotation(15),
    #torchvision.transforms.v2.ColorJitter(brightness=
    transforms.ToTensor(),
    RandomFractionalCrop(min_crop_scale=0.7, max_crop_scale=0.9, seed=69),
])

_DATALOADER_KWARGS = {"num_workers": os.cpu_count(), "prefetch_factor": 2}

class MapillaryDatasetBase(Dataset):
    """Shared operations between CNN and Perceiver dataloader.

    Images from the dataset are loaded and augmented. Every image flowing out of this dataset will
    generally have a different size, so appropriate resizing measures must be taken before
    batching. These are substantially different for the CNN and Perceiver variants.
    """

    def __init__(self, train: bool):
        """Initialize."""
        if train:
            annotation_dict = read_annotation(TRAIN_ANNOTATION_LIST_PATH)
        else:
            annotation_dict = read_annotation(EVAL_ANNOTATION_LIST_PATH)
        self._train = train
        image_paths = []
        annotations = []

        # The class list is made from the set of annotations if it does not already exist.
        class_list = mapillary_class_list()

        # All patch paths are stored with their respective annotation.
        for image_path, class_name in annotation_dict.items():

            # The annotation is stored as a list of size 1.
            # TODO: Fix this.
            class_name = class_name[0]
            image_paths.append(image_path)
            annotations.append(class_list.index(class_name))
        self._image_paths = TorchList(image_paths)
        self._annotations = TorchList(annotations)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index):

        # Read and augment the image.
        with Image.open(self._image_paths[index]) as orig_image:
            orig_image = orig_image.convert("RGB")

            if self._train:
                image = _TRAIN_TRANSFORMS(orig_image)
            else:
                image = _EVAL_TRANSFORMS(orig_image)

            # Some small images will not be contiguous when reaching this point, and this can cause
            # trouble with view operations later on.
            image = image.contiguous()
            return image, self._annotations[index]


def _resize_im(im: torch.Tensor, out_size: Tuple):
    """Resize a tensor image.

    Note that the torchvision.transforms.Resize class expects a PIL image, but we are already
    operating on tensors before we want to do the resize.
    """
    # There is currently a bug in torch that does not let me enable antialias.
    # see https://github.com/pytorch/pytorch/issues/113445
    # TODO: Update when this is fixed.
    return torch.nn.functional.interpolate(im.view(1, *im.shape), size=out_size, mode="bilinear", antialias=False)[0]


class MapillaryDatasetPerceiver(MapillaryDatasetBase):
    """Dataset for training Perceiver.

    A maximum byte tensor size is specified. Images will be flattened first. If they exceed the
    maximum byte tensor size, they will be resized with preservation of aspect ratio to the
    closest size smaller or equal to the maximum tensor size before flattening. If they are smaller
    than the maximum size they can be padded for batched training or be unpadded for inference.
    """
    def __init__(self, max_size: int =40000, **kwargs):
        """Initialize.

        Args:
            max_size: Maximum byte tensor size.
        """
        super().__init__(**kwargs)
        self._max_size = max_size

    def _process_im(self, im: torch.Tensor):
        """Process an image.
    
        See class header for description.
        """
        im_size = im.shape[1]*im.shape[2]
        # If the image is too big we need to resize it, but we want to keep the aspect ratio.
        # Padding might still be needed since we might not end up exactly at the maximum tensor
        # size.
        if im_size > self._max_size:
            scale_factor = math.sqrt(self._max_size/im_size)
            out_size = int(im.shape[1]*scale_factor), int(im.shape[2]*scale_factor)
            im = _resize_im(im, out_size)

        # If the image is small enough we can flatten it and pad it to be fed to Perceiver.
        # We want to return the height and width, but we must recompute them here in case we
        # downsampled an image.
        h, w = im.shape[1:]
        im_size = h*w
        im = im.view(im.shape[0], im_size)
        if im_size == self._max_size:
            return im, h, w

        diff = self._max_size - im.shape[1]
        pad = torch.full([im.shape[0], diff], -1)
        im = torch.cat([im, pad], dim=-1)
        return im, h, w

    def __getitem__(self, index: int):
        """Load an image and do the flattening preprocessing.

        In addition to the image and the annotation we return the height and width so that we can
        use this for positional encodings.

        The output looks like:  (im, h, w), annotation
        """
        im, anno = super().__getitem__(index)
        
        orig_h, orig_w = im.shape[1:]

        im, sh, sw = self._process_im(im)

        # TODO Check if we want to do this reshape in the model instead.
        # The channel dimension must be last for running the transformer.
        im = torch.transpose(im, 0, 1)

        # Generate positional encodings here since arange can not be vectorized on GPU in the model
        # Positional encoding.
        y_pos = torch.arange(sw, dtype=torch.float32).unsqueeze(0)
        x_pos = torch.arange(sh, dtype=torch.float32).unsqueeze(1)
        y_pos = y_pos.repeat(sh, 1)
        x_pos = x_pos.repeat(1, sw)

        y_pos = y_pos.view(sh*sw, 1)
        x_pos = x_pos.view(sh*sw, 1)

        pe = torch.cat([x_pos, y_pos], dim=-1)

        n_empty = im.shape[0] - sh*sw
        pe_empty =  torch.zeros(n_empty, 2)
        pe = torch.cat([pe, pe_empty], dim=0)
        return (im, pe, orig_h, orig_w), anno


def make_dataloader(dataset, batch_size, train):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train, **_DATALOADER_KWARGS)

def get_perceiver_dataloader(batch_size: int, train: bool, max_size: int):
    dset = MapillaryDatasetPerceiver(max_size=max_size, train=train)
    return make_dataloader(dataset=dset, batch_size=batch_size, train=train)


class MapillaryDatasetCNN(MapillaryDatasetBase):
    """Dataset for training CNN models.

    Each image will be resized to the same size.
    """
    def __init__(self, im_size: Tuple, **kwargs):
        """Initialize.
        
        Args:
            im_size: Output image size.
        """
        super().__init__(**kwargs)
        def _resize_op(im):
            return _resize_im(im, im_size)

        self._resize_op = _resize_op

    def __getitem__(self, index):
        """Load an image and resize it."""
        im, anno = super().__getitem__(index)
        h, w = im.shape[1:]
        return (self._resize_op(im), h, w), anno

def get_cnn_dataloader(batch_size: int, train: bool, im_size: Tuple):
    dset = MapillaryDatasetCNN(im_size=im_size, train=train)
    return make_dataloader(dataset=dset, batch_size=batch_size, train=train)


if __name__ == "__main__":

    # Visualize the Perceiver dataset and CNN datasets.
    ds_p = MapillaryDatasetPerceiver(2500, train=True)

    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3

    ds_p = iter(ds_p)
    for i in range(1, columns*rows +1):
        im, h, w= next(ds_p)[0]
        im = im[:,:h*w]
        im = einops.rearrange(im, "c (h w) -> h w c", h=h, w=w)
        fig.add_subplot(rows, columns, i)
        plt.imshow(im)
    plt.show()
    plt.clf()
    

    ds_c = MapillaryDatasetCNN((50, 50), train=True)
    ds_c = iter(ds_c)
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows +1):
        im = next(ds_c)[0]
        im = einops.rearrange(im, "c h w -> h w c")
        fig.add_subplot(rows, columns, i)
        plt.imshow(im)
    plt.show()
