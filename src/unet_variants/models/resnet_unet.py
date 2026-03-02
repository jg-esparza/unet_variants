from torch.nn import Module

from unet_variants.models.components.unet.resnet import ResNet
from unet_variants.models.components.unet.decoder import Decoder
from unet_variants.models.components.modules import SegmentationHead

class ResNetUnet(Module):
    def __init__(self, config):
        super(ResNetUnet, self).__init__()
        self.encoder = ResNet()
        self.decoder = Decoder(config, conv_more=False)
        self.segmentation_head = SegmentationHead(in_channels=config.decoder_channels[-1],
                                                  out_channels=config.out_channels,
                                                  kernel_size=3,
                                                  upsampling=2)

    def forward(self, x):
        x, features = self.encoder(x)
        x = self.decoder(x, features)
        return self.segmentation_head(x)

    def load_from(self, path=None):
        if path is not None:
            print('Define load function')
        else:
            print("none pretrain")

