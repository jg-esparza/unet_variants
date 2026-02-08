from torch.nn import Module

from unet_variants.models.unet_components.components import SegmentationHead
from unet_variants.models.unet_components.encoder import Encoder
from unet_variants.models.unet_components.decoder import Decoder

class UNet(Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, in_channels=config.hidden_layers)
        self.segmentation_head = SegmentationHead(in_channels=config.decoder_channels[-1],
                                                  out_channels=config.out_channels,
                                                  kernel_size=3,
                                                  upsampling=1)

    def forward(self, x):
        x, features = self.encoder(x)
        x = self.decoder(x, features)
        return self.segmentation_head(x)

    def load_from(self, load_ckpt_path):
        if load_ckpt_path is not None:
            print('Define load function')
        else:
            print("none pretrain")
