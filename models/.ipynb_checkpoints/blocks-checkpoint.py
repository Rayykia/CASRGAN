"""The basic blocks used in model construction.

Author: Ruichen Deng
Date: 3/16/2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union, Callable
from torch import Tensor

from . import utils




def default_conv(
        input_nc: int,
        output_nc: int, 
        kernel_size: int,
        bias: bool = True
) -> nn.Module:
    """Return a default convolution layer that do not change the feature size.
    
    Parameters
    ----------
    input_nc (int)
        -- the number of channels in the input feature
    output_nc (int)
        -- the number of channels in the output feature
    kernel_size (int)
        -- the kernel size
    bias (bool)
        -- use bias
    """
    return nn.Conv2d(
        input_nc, output_nc, kernel_size, padding=(kernel_size//2), bias=bias
    )




class ResBlock(nn.Module):
    def __init__(
            self, 
            conv_layer: Callable[...,nn.Conv2d],
            ngf: int,
            kernel_size: int,
            bias: bool = True,
            norm_layer: str = 'none',
            act_layer: nn.Module = nn.ReLU(),
            res_scale: float = 1.0,
            *args, **kwargs
    ) -> None:
        """Residual Block for EDSR.
        
        Parameters
        ----------
        conv_layer (Callable[...,nn.Conv2d])
            -- the convolution layer used
        ngf (int)
            -- the number of filters within one convolution layer
        kernel_size (int)
            -- the kernel size
        bias (bool)
            -- use bias `default: True`
        norm_layer (str)
            -- the normalization layer: [batch_norm | instance_norm | none] `default: 'none'`
        act_layer (nn.Module)
            -- activation function `default: nn.ReLU()`
        res_scale (float)
            -- residual scaling factor, 0.1 was used in EDSR `default: 1`
        """
        super().__init__(*args, **kwargs)

        modules = []

        for i in range(2):
            modules.append(conv_layer(ngf, ngf, kernel_size, bias=bias))
            if norm_layer != 'none':
                modules.append(utils.get_norm_layer(norm_layer))
            if i == 0:
                modules.append(act_layer)
        
        self.model = nn.Sequential(*modules)
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        """Residual forward + Identity forward"""
        residule = self.model(x).mul(self.res_scale)
        residule += x
        return residule




class Upsampler(nn.Module):
    def __init__(
            self,
            conv_layer: nn.Conv2d,
            scale: int,
            ngf: int,
            norm_layer: str = 'none',
            act: str =  None,
            bias: bool = False
    ) -> None:
        """Upsampler block for EDSR.
        
        Parameters
        ----------
        conv_layer (nn.Conv2d)
            -- the convolution layer used
        scale (int)
            -- upsampling scale 
        ngf (int)
            -- the number of filters within one convolution layer
        norm_layer (str)
            -- the normalization layer: [batch_norm | instance_norm | none] `default: 'none'`    
        bias (bool)
            -- use bias `default: True`
        act (str)
            -- activation layer [relu | prelu]
        """
        super().__init__()
        modules = []
        
        if (scale & (scale - 1)) == 0:
            for _ in range(int(np.log2(scale))):
                modules.append(conv_layer(ngf, 4*ngf, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if norm_layer != 'none':
                    modules.append(utils.get_norm_layer(norm_layer))
                if act == 'relu':
                    modules.append(nn.ReLU(inplace=True))
                if act == 'prelu':
                    modules.append(nn.PReLU(ngf))
        elif scale == 3:
            modules.append(conv_layer(ngf, 9 * ngf, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if norm_layer != 'none':
                modules.append(utils.get_norm_layer(norm_layer))
            if act == 'relu':
                modules.append(nn.ReLU(True))
            elif act == 'prelu':
                modules.append(nn.PReLU(ngf))
        else:
            raise NotImplementedError('This model is only implement to operate on [x2|x3|x4] scale SR')
        
        self.model = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward."""
        return self.model(x)




class _RDB_conv(nn.Module):
    def __init__(
            self,
            input_nc: int,
            growth_rate: int,
            kernel_size: int
    ) -> None:
        """Convolution layer used in the Resudyke Debse Block in RDN.

        nn.Conv2d + nn.ReLU
        
        Parameters
        ----------
        input_nc (int)
            -- # of input channel of the layer
        growth_rate (int)
            -- # of output channels of the layer
        kernel_size (int)
            -- kernel size of the convolution layer, `default: 3`
        """
        super().__init__()
        self.conv = nn.Sequential(
            default_conv(input_nc=input_nc, output_nc=growth_rate, kernel_size=kernel_size),
            nn.ReLU()
        )

    def forward(self, x: Tensor):
        """Concatenate the input and the output of the conv+relu layer as the final output."""
        out = self.conv(x)
        return torch.cat((x, out), dim=1)




class ResidualDenseBlock(nn.Module):
    def __init__(
            self, 
            growth_rate0: int,
            growth_rate: int,
            n_conv_layers: int,
            k_size: int
    ) -> None:
        """Residual Dense Block (RDB) in RDN.

        Parameters
        ----------
        growth_rate_0 (int)
            -- # of input and output channels of the RDB
        growth_rate (int)
            -- # of the output channels of the inner convolution layers
        n_conv_layers (int)
            -- # of convlution layers used within an RDB, referenced as 'C' in the original paper
        k_size (int)
            -- kernel size of the inner convolution layers
        """
        super().__init__()

        conv_layers = []

        for c in range(n_conv_layers):
            conv_layers.append(
                _RDB_conv(
                    input_nc= growth_rate0 + c * growth_rate,
                    growth_rate=growth_rate,
                    kernel_size = k_size
                )
            )
        self.convs = nn.Sequential(*conv_layers)

        # local feature fusion
        self.LFF = nn.Conv2d(
            in_channels=growth_rate0 + n_conv_layers * growth_rate,
            out_channels=growth_rate0,
            kernel_size=1,
            padding=0,
            stride=1
        )

    def forward(self, x: Tensor):
        """dense convolution + LFF + LRL
        
        ..notes:
            LFF: Local Feature Fusion
            LRL: Local Residual Learning
        """
        return self.LFF(self.convs(x)) + x
    



class _DRRNResidualUnit(nn.Module):
    def __init__(
            self,
            ngf: int,
            ksize: int
    ) -> None:
        """Residule unit in the Deep Recursive Residule Network (DRRN).

        Parameters
        ----------
        ngf (int)
            -- # of filters in the conv-layers
        ksize (int)
            -- kernel size
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf, ksize, stride=1, padding=ksize//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf, ksize, stride=1, padding=ksize//2)
        )

    def forward(
            self,
            h_0: Tensor,
            h_in: Tensor
    ) -> Tensor:
        """residule forward with shared identity
        
        Parameters
        ----------
        h_0 (Tensor)
            -- output of the 1st conv-layer within the recursive block
        h_in (Tensor)
            -- input of the residule unit
        """
        return self.conv(h_in) + h_0




class RecursiveBlock(nn.Module):
    def __init__(
            self, 
            n_res: int,
            input_nc: int, 
            ngf: int,
            ksize: int
    ) -> None:
        """Recursive block (RB) in the Deep Recursive Residule Network (DRRN).

        Parameters
        ----------
        n_res (int)
            -- # of residule units within one recursive block, referenced as 'U' in the original paper
        input_nc (int)
            -- # of input channels
        ngf (int)
            -- # of filters
        ksize (int)
            -- kernel size
        
        .. notes:
            All residual units within one recursive block share weights.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(input_nc, ngf, ksize, stride=1, padding=ksize//2)
        )
        self.residule_unit = _DRRNResidualUnit(ngf, ksize)

        self.n_res = n_res

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        h_0 = self.conv1(x)
        x = h_0

        for _ in range(self.n_res):
            x = self.residule_unit(h_0, x)

        return x