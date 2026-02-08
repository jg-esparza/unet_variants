from torch.nn import Module, ModuleList

from unet_variants.models.components.unet_components.components import Conv2dReLU, DownSample

class Encoder(Module):
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

import torch
import ml_collections

network_config = ml_collections.ConfigDict()
network_config.encoder_in_channels = [3, 64, 128, 256]
network_config.encoder_out_channels = [64, 128, 256, 512]
network_config.n_skip = 4
network_config.hidden_layers = 1024

if __name__ == '__main__':
    encoder = Encoder(network_config)
    # print(encoder)
    input_encoder = torch.randn(1, 3, 256, 256)
    output, out_features = encoder(input_encoder)
    print(output.size())
    print(len(out_features))
    for feature in out_features:
         print(feature.size())
    # print(features.size())
