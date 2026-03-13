from torch.nn import Module, ModuleList

from unet_variants.models.components.modules import Conv2dReLU, DownSample
from unet_variants.models.components.unet.resnet import ResNet

class EncoderBase(Module):
    def __init__(self, config):
        super().__init__()
        config.encoder_in_channels.insert(0, config.in_channels)
        in_channels = config.encoder_in_channels
        out_channels = config.encoder_out_channels
        hidden_layers = config.hidden_layers
        depth = len(in_channels)
        self.skip_channels = []
        if config.n_skip != 0 and config.n_skip <= depth:
            for i in range(depth, depth - config.n_skip, -1):
                self.skip_channels.append(i)

        blocks = [
            DownSample(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)
        ]
        self.blocks = ModuleList(blocks)
        self.conv_more = Conv2dReLU(in_channels=out_channels[-1], out_channels=hidden_layers,
                                    kernel_size=3, padding=1, use_batchnorm=True)

    def forward(self, x):
        features = []
        for i, encoder_block in enumerate(self.blocks):
            x, skip = encoder_block(x)
            if i+1 in self.skip_channels:
                features.append(skip)
        x = self.conv_more(x)
        return x, features[::-1]

class Encoder(Module):
    def __init__(self, config):
        super().__init__()
        encoder_name = config.encoder
        if encoder_name == "base":
            self.encoder = EncoderBase(config)
        elif "resnet" in encoder_name:
            self.encoder = ResNet(encoder_name)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.encoder(x)