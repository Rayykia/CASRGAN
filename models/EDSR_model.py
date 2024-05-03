"""The EDSR model.

Date: 3/16/2024
"""
from torch import nn

from argparse import Namespace

from . import blocks
from typing import Callable
from torch import Tensor




class EDSRModel(nn.Module):
    def __init__(
            self, 
            opt: Namespace,
            conv_layer: Callable[..., nn.Conv2d] = blocks.default_conv,
            *args, 
            **kwargs
    ) -> None:
        """The EDSR model.
        
        Parameters:
            opt (Namespace)                             -- options
            conv_layer (Callable[..., nn.Conv2d])       -- fucntion to get a convolution layer
        """
        super().__init__(*args, **kwargs)
        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        kernel_size=3
        scale=opt.scale
        res_scale=0.1
        act = nn.ReLU(True)

        model_1 = [conv_layer(1, n_feats, kernel_size)]

        model_2 = [
            blocks.ResBlock(
                conv_layer, n_feats, kernel_size, act_layer=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        model_2.append(conv_layer(n_feats, n_feats, kernel_size))

        model_3 = [
            blocks.Upsampler(conv_layer, scale, n_feats, act='none'),
            conv_layer(n_feats, 1, kernel_size)
        ]

        self.model_1 = nn.Sequential(*model_1)
        self.model_2 = nn.Sequential(*model_2)
        self.model_3 = nn.Sequential(*model_3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model_1(x)
        x = self.model_2(x) + x
        x = self.model_3(x)
        return x