import torch
import random
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import einops

class RandomFractionalCrop:
    """Random fractional cropping operator.

    A random cropping fraction will be determined. Then this fraction of the image will be cropped
    out, and the original aspect ratio will be preserved. The part of the image being cropped after
    the scale is determined is uniformly sampled across the image.
    """
    def __init__(self, min_crop_scale: int, max_crop_scale: int, seed=-1):
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

    def _random_size(self, h, w):
        min_h = int(h*self._min_scale)
        max_h = int(h*self._max_scale)
        min_w = int(w*self._min_scale)
        max_w = int(w*self._max_scale)

        crop_h = random.randint(min_h, max_h)
        crop_w = random.randint(min_w, max_w)
        return crop_h, crop_w

    def __call__(self, x):
        """Randomly crop an image tensor."""
        h, w = x.shape[1:]
        
        rh, rw = self._random_size(h, w)
        print(rw, rh)

        max_corner = (h - rh, w - rw)
        rx_corner = random.randint(0, max_corner[0])
        ry_corner = random.randint(0, max_corner[1])
        
        return torchvision.transforms.functional.crop(x, rx_corner, ry_corner, rh, rw)

    def __repr__(self):
        return """Aspect ratio preserving random crop for any image size."""


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
