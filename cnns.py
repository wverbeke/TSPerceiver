import torch
from torch import nn

class ResnetBlockB(nn.Module):
    """Type B building block for resnet as defined in https://arxiv.org/pdf/1512.03385.pdf."""

    def __init__(self, in_channels: int, out_channels: int, downsample_ratio: int):
        super().__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(downsample_ratio, downsample_ratio), padding=1)
        self._bn_1 = nn.BatchNorm2d(out_channels)
        self._conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self._bn_2 = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU()
        if downsample_ratio != 1:
            self._skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(downsample_ratio, downsample_ratio))
            self._bn_skip = nn.BatchNorm2d(out_channels)
        else:
            self._skip = (lambda x: x)
            self._bn_skip = (lambda x: x)

    def forward(self, x):
        conv_path_out = self._relu(self._bn_1(self._conv_1(x)))
        conv_path_out = self._bn_2(self._conv_2(conv_path_out))
        skip_path_out = self._bn_skip(self._skip(x))
        return self._relu(torch.add(conv_path_out, skip_path_out))


class Resnet18(nn.Module):
    """Resnet 18 as defined in https://arxiv.org/pdf/1512.03385.pdf."""

    def __init__(self, in_channels: int = 3, channel_multiplier: float = 1.0):
        super().__init__()
        channels = int(64*channel_multiplier)
        self._init_conv = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self._init_bn = nn.BatchNorm2d(channels)
        self._relu = nn.ReLU()
        self._init_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        
        self._resnet_body = nn.Sequential()

        # Stage 1
        self._resnet_body.append(ResnetBlockB(in_channels=channels, out_channels=channels, downsample_ratio=1))
        self._resnet_body.append(ResnetBlockB(in_channels=channels, out_channels=channels, downsample_ratio=1))

        # Stage 2
        self._resnet_body.append(ResnetBlockB(in_channels=channels, out_channels=channels*2, downsample_ratio=2))
        self._resnet_body.append(ResnetBlockB(in_channels=channels*2, out_channels=channels*2, downsample_ratio=1))

        # Stage 3
        self._resnet_body.append(ResnetBlockB(in_channels=channels*2, out_channels=channels*4, downsample_ratio=2))
        self._resnet_body.append(ResnetBlockB(in_channels=channels*4, out_channels=channels*4, downsample_ratio=1))

        # Stage 4
        self._resnet_body.append(ResnetBlockB(in_channels=channels*4, out_channels=channels*8, downsample_ratio=2))
        self._resnet_body.append(ResnetBlockB(in_channels=channels*8, out_channels=channels*8, downsample_ratio=1))

        # Store then number of output channels for later use.
        self._out_channels = (channels*8)

    def forward(self, x):
        init_out = self._init_pool(self._relu(self._init_bn(self._init_conv(x))))
        return self._resnet_body(init_out)

    @property
    def out_channels(self):
        return self._out_channels


class SimpleCNN(nn.Module):
    """A simple CNN with relatively few parameters."""

    @staticmethod
    def _build_conv(in_channels: int, out_channels: int, downsample_ratio: int):
        conv_element = nn.Sequential()
        conv_element.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(downsample_ratio, downsample_ratio), padding=1))
        conv_element.append(nn.BatchNorm2d(out_channels))
        conv_element.append(nn.ReLU())
        return conv_element

    def __init__(self, in_channels: int = 3, channel_multiplier: float = 1.0):
        super().__init__()
        channels = int(16*channel_multiplier)
        self._body = nn.Sequential()
        
        # Stage 1
        self._body.append(SimpleCNN._build_conv(in_channels=in_channels, out_channels=channels, downsample_ratio=1))
        self._body.append(SimpleCNN._build_conv(in_channels=channels, out_channels=channels, downsample_ratio=1))
        self._body.append(SimpleCNN._build_conv(in_channels=channels, out_channels=channels*2, downsample_ratio=2))
        self._body.append(SimpleCNN._build_conv(in_channels=channels*2, out_channels=channels*2, downsample_ratio=1))
        self._body.append(SimpleCNN._build_conv(in_channels=channels*2, out_channels=channels*4, downsample_ratio=2))
        self._body.append(SimpleCNN._build_conv(in_channels=channels*4, out_channels=channels*4, downsample_ratio=1))
        self._body.append(SimpleCNN._build_conv(in_channels=channels*4, out_channels=channels*8, downsample_ratio=2))

        self._out_channels = (channels*8)

    def forward(self, x):
        return self._body(x)

    @property
    def out_channels(self):
        return self._out_channels


class CNNClassifier(nn.Module):
    """Wrap a model backbone to become a classifier.

    The CrossEntropy loss in pytorch expects raw logits, so softmax should not be used with those
    loss functions.
    """
    def __init__(self, body: nn.Module, num_classes: int, softmax: bool = False):
        super().__init__()
        self._body = body
        def _global_pool(tensor):
            return torch.mean(tensor, dim=[2, 3])
        self._global_pool = _global_pool
        self._dense = nn.Linear(in_features=body.out_channels, out_features=num_classes, bias=True)
        self._softmax = nn.Softmax(dim=1) if softmax else lambda x: x

    def forward(self, x: torch.Tensor, h: int, w: int):
        x = self._body(x)
        x = self._global_pool(x)
        x = self._dense(x)
        return self._softmax(x)


# Test the model building code by building ONNX files that can be inspected easily with netron.
if __name__ == "__main__":
    resnet = Classifier(Resnet18(), 10)
    simple_cnn = Classifier(SimpleCNN(), 10)

    # An tensor must be use to specify the input size for the ONNX model.
    i = torch.rand(1, 3, 64, 64)
    torch.onnx.export(resnet, i, "resnet.onnx")
    torch.onnx.export(simple_cnn, i, "simple_cnn.onnx")
