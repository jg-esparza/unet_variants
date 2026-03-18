from omegaconf import DictConfig

from torch.nn import Module

from models.cnn.components.encoder import Encoder
from models.cnn.components.decoder import Decoder
from models.cnn.components.modules import SegmentationHead

class UNet(Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg.unet_config
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)
        self.segmentation_head = SegmentationHead(in_channels=self.cfg.decoder_channels[-1],
                                                  out_channels=self.cfg.out_channels,
                                                  kernel_size=self.cfg.segmentation_head.kernel_size,
                                                  upsampling=self.cfg.segmentation_head.upsampling)

    def forward(self, x):
        x, features = self.encoder(x)
        x = self.decoder(x, features)
        return self.segmentation_head(x)

    def load_from(self, path=None):
        if path is not None:
            print('Define load function')
        else:
            print("none pretrain")
