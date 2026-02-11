import torch
import torch.nn as nn

from unet_variants.models.components.unet_components.resnet import ResNet
from unet_variants.models.components.unet_components.components import SegmentationHead
from unet_variants.models.components.unet_components.decoder import Decoder

class ResNetUnet(nn.Module):
    def __init__(self, config):
        super(ResNetUnet, self).__init__()
        self.n_channels = config.in_channels
        self.n_classes = config.out_channels
        self.encoder = ResNet()
        self.decoder = Decoder(config, in_channels=config.hidden_layers, conv_more=False)
        self.segmentation_head = SegmentationHead(in_channels=config.decoder_channels[-1],
                                                  out_channels=config.out_channels,
                                                  kernel_size=3,
                                                  upsampling=2)

    def forward(self, x):
        x, features = self.encoder(x)
        x = self.decoder(x, features)
        return self.segmentation_head(x)
