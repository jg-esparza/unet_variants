from torch.nn import Module

from unet_variants.models.components.unet.encoder import Encoder
from unet_variants.models.components.unet.decoder import Decoder
from unet_variants.models.components.modules import SegmentationHead

class UNet(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.unet_config
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.segmentation_head = SegmentationHead(in_channels=self.config.decoder_channels[-1],
                                                  out_channels=self.config.out_channels,
                                                  kernel_size=3,
                                                  upsampling=1)

    def forward(self, x):
        x, features = self.encoder(x)
        x = self.decoder(x, features)
        return self.segmentation_head(x)

    def load_from(self, path=None):
        if path is not None:
            print('Define load function')
        else:
            print("none pretrain")
