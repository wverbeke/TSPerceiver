"""Simple class representing a bounding box."""
import math

import torch
from torchvision import transforms


class Bbox:

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):

        # The maximum will be used as a one-off-the-end index while the minimum is the first index.
        self._xmin = int(math.floor(xmin))
        self._xmax = int(math.ceil(xmax)) + 1
        self._ymin = int(math.floor(ymin))
        self._ymax = int(math.ceil(ymax)) + 1

    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @property
    def ymin(self):
        return self._ymin

    @property
    def ymax(self):
        return self._ymax

    @property
    def width(self):
        return (self.xmax - self.xmin)

    @property
    def height(self):
        return (self.ymax - self.ymin)

    @staticmethod
    def from_dict(bbox_dict):
        return Bbox(bbox_dict["xmin"], bbox_dict["xmax"], bbox_dict["ymin"], bbox_dict["ymax"])

    
    def crop_from_image(self, image_tensor, padding_fraction=0.0):
        """Extract the area corresponding to the bounding box from an image tensor."""
        # In pytorch the top left corner of the image is at (0, 0).
        # The y-axis represents the height and the x-axis the width.

        if padding_fraction > 0:
            x_offset = int(math.ceil(self.width*padding_fraction/2))
            start_x = self.xmin - x_offset
            width = self.width + 2*x_offset
            
            y_offset = int(math.ceil(self.height*padding_fraction/2))
            start_y =self.ymin - y_offset
            height = self.height + 2*y_offset
    
            image_height, image_width = image_tensor.shape[2:]
            xmin_pad, xmax_pad, ymin_pad, ymax_pad = 0, 0, 0, 0
            if start_x < 0:
                xmin_pad= (0 - start_x)
                start_x = 0
            if (start_x + width) > image_width:
                xmax_pad = start_x + width - image_width
                width = (image_width - start_x)
            if start_y < 0:
                ymin_pad = (0 - start_y)
                start_y = 0
            if (start_y + height) > image_height:
                ymax_pad = start_y + height - image_height
                height = (image_height - start_y)

            patch = transforms.functional.crop(image_tensor, top=start_y, left=start_x, height=height, width=width)
            if any(i > 0 for i in [xmin_pad, xmax_pad, ymin_pad, ymax_pad]):
                patch = torch.nn.functional.pad(patch, (xmin_pad, xmax_pad, ymin_pad, ymax_pad), mode="reflect")
            return patch


        patch = transforms.functional.crop(image_tensor, top=self.ymin, left=self.xmin, height=self.height, width=self.width)
        return patch
