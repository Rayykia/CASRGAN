"""The RDN model.

Date: 3/23/2024
"""
import torch
from torch import nn

from argparse import Namespace

from . import blocks
from torch import Tensor


class RDNModel(nn.Module):
    def __init__(
            self, 
            opt: Namespace
    ) -> None:
        """The RDN model.
        
        Parameters:
            opt (Namespace)         -- options
        """
        super().__init__()

        growth_rate0 = opt.growth_rate0
        growth_rate = opt.growth_rate
        rdb_depth = opt.rdb_depth
        self.rdn_depth = opt.rdn_depth
        ksize = opt.rdn_ksize
        scale = opt.scale
        input_nc = opt.input_nc
        output_nc = opt.output_nc

        # SFE: Shallow Feature Extraction
        self.SFE1 = nn.Conv2d(
            in_channels=input_nc,
            out_channels=growth_rate0,
            kernel_size=ksize,
            padding=ksize//2,
            stride=1
        )
        self.SFE2 = nn.Conv2d(
            in_channels=growth_rate0,
            out_channels=growth_rate0,
            kernel_size=ksize,
            padding=ksize//2,
            stride=1
        )

        # RDBs + DFF:Demse Feature Fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.rdn_depth):
            self.RDBs.append(
                blocks.ResidualDenseBlock(
                    growth_rate0=growth_rate0,
                    growth_rate=growth_rate,
                    n_conv_layers=rdb_depth,
                    k_size=ksize
                )
            )
        
        # GFF: Global Feature Fusion
        self.GFF = nn.Sequential(
            nn.Conv2d(
                in_channels=self.rdn_depth * growth_rate0, 
                out_channels=growth_rate0,
                kernel_size=1, padding=0, stride=1
            ),
            nn.Conv2d(
                growth_rate0, growth_rate0,
                kernel_size=ksize, padding=ksize//2, stride=1
            )
        )

        # Upscale
        if scale == 2 or scale == 3:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    growth_rate0, growth_rate*scale*scale, kernel_size=ksize, padding=ksize//2,stride=1
                ),
                nn.PixelShuffle(scale),
                nn.Conv2d(
                    growth_rate, output_nc, ksize, padding=ksize//2, stride=1
                )
            )
        elif scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    growth_rate0, growth_rate*4, kernel_size=ksize, padding=ksize//2,stride=1
                ),
                nn.PixelShuffle(2),
                nn.Conv2d(
                    growth_rate, growth_rate*4, ksize, padding=ksize//2, stride=1
                ),
                nn.PixelShuffle(2),
                nn.Conv2d(
                    growth_rate, output_nc, ksize, padding=ksize//2, stride=1
                )
            )
        else:
            raise ValueError('scale only supprt [2|3|4] for the moment')
        

    def forward(self, x: Tensor):
        """forward: dense forward + GRL + upscale"""
        f_m1 = self.SFE1(x)
        x = self.SFE2(f_m1)

        RDBs_out = []
        for i in range(self.rdn_depth):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, dim=1))
        x += f_m1
        return self.upscale(x)