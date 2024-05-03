"""The DRRN model.

Date: 3/24/2024
"""
from torch import nn

from argparse import Namespace

from . import blocks
from typing import Callable
from torch import Tensor




class DRRNModel(nn.Module):
    def __init__(
            self,
            opt: Namespace
    ) -> None:
        super().__init__()
        B = opt.drrn_rb_depth
        U = opt.drrn_ru_depth
        ksize = opt.drrn_ksize
        ngf = opt.drrn_ngf
        input_nc = opt.input_nc
        output_nc = opt.output_nc


        rbs = []
        for i in range(B):
            rbs.append(blocks.RecursiveBlock(
                n_res= U, input_nc=input_nc if i == 0  else ngf, ngf=ngf, ksize=ksize
            ))
        self.rbs = nn.Sequential(*rbs)

        self.recon = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_nc, kernel_size=ksize, padding=ksize//2)
        )

    def forward(
            self, 
            x: Tensor
    ) -> Tensor:
        """Residule fdorward for DRRN."""
        identity = x
        x = self.rbs(x)
        x = self.recon(x)
        return identity + x