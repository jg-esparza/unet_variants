from omegaconf import DictConfig

import copy
import logging

import torch
import torch.nn as nn

from models.transformers.components.swinunet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SwinUnet, self).__init__()
        self.cfg = cfg
        self.swin_unet = SwinTransformerSys(img_size=cfg.image_size,
                                patch_size=cfg.swin.patch_size,
                                in_chans=cfg.in_channels,
                                num_classes=cfg.out_channels,
                                embed_dim=cfg.swin.embed_dim,
                                depths=cfg.swin.depths,
                                depths_decoder=cfg.swin.decoder_depths,
                                num_heads=cfg.swin.num_heads,
                                window_size=cfg.swin.window_size,
                                mlp_ratio=cfg.swin.mlp_ratio,
                                qkv_bias=cfg.swin.qkv_bias,
                                qk_scale=cfg.swin.qk_scale,
                                ape=cfg.swin.ape,
                                patch_norm=cfg.swin.patch_norm,
                                drop_rate=cfg.drop_rate,
                                drop_path_rate=cfg.drop_path_rate,
                                use_checkpoint=cfg.use_checkpoint)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, path=None):
        pretrained_path = self.cfg.pretrained_ckpt if path is None else path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")