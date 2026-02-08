from torch.nn import Module, ModuleList

from unet_variants.models.components.unet_components.components import Conv2dReLU, UpSample

class Decoder(Module):
    def __init__(self, config, in_channels=1024, head_channels = 512, conv_more=True):
        super().__init__()
        self.config = config
        head_channels = head_channels
        if conv_more:
            self.conv_more = Conv2dReLU(in_channels,
                                        head_channels,
                                        kernel_size=3,
                                        padding=1,
                                        use_batchnorm=True)
        else:
            self.conv_more = None
            head_channels = in_channels
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            UpSample(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        if self.conv_more is None:
            x = hidden_states
        else:
            x = self.conv_more(hidden_states)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

import torch
import ml_collections

network_config = ml_collections.ConfigDict()
network_config.decoder_channels = (256, 128, 64, 16)
network_config.skip_channels = [512, 256, 128, 64]
network_config.n_skip = 4

if __name__ == '__main__':
    input_decoder = torch.randn(1, 1024, 16, 16)
    features = [torch.randn(1, 512, 32, 32),
                torch.randn(1, 256, 64, 64),
                torch.randn(1, 128, 128, 128),
                torch.randn(1, 64, 256, 256)]
    decoder = Decoder(network_config)
    output = decoder(input_decoder, features)
    print(output.size())
    # print(decoder)

