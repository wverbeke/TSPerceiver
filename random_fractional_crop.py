import torch
import random
import math
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import einops
from mapillary_data_loader.preproc_mapillary import PADDING_FRACTION

class RandomFractionalCrop:
    """Random fractional cropping operator.

    A random cropping fraction will be determined. Then this fraction of the image will be cropped
    out, and the original aspect ratio will be preserved. The part of the image being cropped after
    the scale is determined is uniformly sampled across the image.
    """
    def __init__(self, min_crop_scale: float, max_crop_scale: float, seed: int = -1):
        """Init.

        Args:
            min_crop_scale: Minimum fraction of the image that a crop will be.
            max_crop_scale: Maximum fraction of the image that a crop will be.
            seed: Random seed.
        """
        self._min_scale = min_crop_scale
        self._max_scale = max_crop_scale
        self._seed = seed
        random.seed(seed)

    def _random_size(self, h: int, w: int):
        min_h = int(h*self._min_scale)
        max_h = int(h*self._max_scale)
        min_w = int(w*self._min_scale)
        max_w = int(w*self._max_scale)

        crop_h = random.randint(min_h, max_h)
        crop_w = random.randint(min_w, max_w)
        return crop_h, crop_w

    def __call__(self, x: torch.Tensor):
        """Randomly crop an image tensor."""
        h, w = x.shape[1:]
        
        rh, rw = self._random_size(h, w)

        max_corner = (h - rh, w - rw)
        rx_corner = random.randint(0, max_corner[0])
        ry_corner = random.randint(0, max_corner[1])
        
        return torchvision.transforms.functional.crop(x, rx_corner, ry_corner, rh, rw)

    def __repr__(self):
        return """Aspect ratio preserving random crop for any image size."""


class FractionalCenterCrop:
    """Fractional center crop operator.

    Used during evaluation to remove the padding added during preprocessing for random training crops.
    """
    def __init__(self, pad_ratio):
        assert pad_ratio > 0.0
        self._pad_ratio = pad_ratio

    def __call__(self, x: torch.Tensor):
        """Center crop an image fractionally."""
        h, w = x.shape[1:]

        # Warning: This will not restore the exact same image as one would get if the padding ratio
        # is 0 in the dataset preprocessing. The reason is that "ceil" and an int conversion is
        # used there, and the ceiling function is not bijective so has no clear inverse.

        # We take the floor here because the true correction factor must at least be as big as
        # (1 + self._pad_ratio) and the ceil function is applied to the left and right when making
        # the padding.
        orig_h = math.floor(h/(1 + self._pad_ratio))
        orig_w = math.floor(w/(1 + self._pad_ratio))

        return torchvision.transforms.functional.center_crop(x, [orig_h, orig_w])



if __name__ == "__main__":
    # Test the operator.
    test_path = "eval_patches/0xqRtdRNdp11h_OMbC7XZw_patch_0.png"
    test_image = Image.open(test_path)
    test_image = ToTensor()(test_image)
    print(test_image.shape)
    
    rs = RandomFractionalCrop(0.7, 0.9)
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for i in range(1, columns*rows +1):
        im = rs(test_image)
        im = einops.rearrange(im, "c h w -> h w c")
        fig.add_subplot(rows, columns, i)
        plt.imshow(im)
    plt.show()
    plt.clf()

    fcc = FractionalCenterCrop(PADDING_FRACTION)
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows +1):
        im = fcc(test_image)
        im = einops.rearrange(im, "c h w -> h w c")
        fig.add_subplot(rows, columns, i)
        plt.imshow(im)
    plt.show()
