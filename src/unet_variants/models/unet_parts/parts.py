import torch
from torch.nn import Conv2d, BatchNorm2d, Identity, Module, ReLU, Sequential, UpsamplingBilinear2d, MaxPool2d

class Conv2dReLU(Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, bias=not use_batchnorm)
        relu = ReLU(inplace=True)
        bn = BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DoubleConv(Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, use_batchnorm=True):
        conv1 = Conv2dReLU(in_channels, out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, use_batchnorm=use_batchnorm)
        conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, use_batchnorm=use_batchnorm)
        super(DoubleConv, self).__init__(conv1, conv2)

class DownSample(Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(DownSample, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=3,
                                      stride=1, padding=1, use_batchnorm=use_batchnorm)

        self.maxpool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.double_conv(x)
        x = self.maxpool(skip)
        return x, skip

class UpSample(Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.double_conv = DoubleConv(in_channels + skip_channels, out_channels,
                                      kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)

class SegmentationHead(Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else Identity()
        activation = ActivationFunction(out_channels)
        super().__init__(conv2d, upsampling, activation)

class ActivationFunction(Module):
    def __init__(self, classes):
        self.activation = torch.sigmoid if classes <= 2 else torch.softmax
        super().__init__()
    def forward(self, x):
        return self.activation(x)

if __name__ == '__main__':
    input_rand = torch.randn(1, 3, 256, 256)
    down_sample = DownSample(in_channels=3, out_channels=64)
    print(down_sample)
    output, out_skip = down_sample(input_rand)
    print(output.shape)
    print(out_skip.shape)