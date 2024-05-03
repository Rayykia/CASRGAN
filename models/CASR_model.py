"""Generator of Super-Resolution GAN with Combined Attnetion (CASRGAN)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import blocks


from argparse import Namespace
from typing import Callable, Optional
from torch import Tensor




class PatchEmbed(nn.Module):
    """Patch Embedding Layer + Linear Projection.
    
    (B, H, W, L) -> (B, H*W/p^2, embed_dim); p: patch_size
    """
    def __init__(
            self, 
            feat_size: int = 96,
            patch_size: int = 4, 
            in_channels: int = 1,
            embed_dim: int = 48, 
            norm_layer: Optional[Callable[..., Tensor]] = None,
            *args, **kwargs
    ) -> None:
        """
        Parameters:
            feat_size (int)                                     -- the input feature size (H, W)
            patch_size (int)                                    -- patch size for embedding
            in_channels (int)                                   -- # of input channels
            embed_dim (int)                                     -- embeded dimention for swin-transformer
            norm_layer (Optional[Callable[..., Tensor]])        -- normalization layer
        """
        super().__init__(*args, **kwargs)
        patches_resolution = feat_size // patch_size

        self.img_size = feat_size
        self.patch_size = patch_size
        self.patchs_resolution = patches_resolution
        self.num_patches = patches_resolution ** 2
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels=in_channels, out_channels=embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )

        if norm_layer is not None:
            self.norm_layer = norm_layer
        else:
            self.norm_layer = None
    
    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        x = self.proj(x).flatten(2).transpose(1,2)  # (B, H*W/n^2, embed_dim)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x




class PatchUnembed(nn.Module):
    """Patch Merging Layer."""
    def __init__(
            self, 
            feat_size: int,
            patch_size: int, 
            embed_dim: int,
            out_channels: int, 
            *args, **kwargs
    ) -> None:
        """
        Parameters:
            feat_size (int)                                     -- the input feature size (H, W)
            patch_size (int)                                    -- patch size for embedding
            in_channels (int)                                   -- # of input channels
            embed_dim (int)                                     -- embeded dimention for swin-transformer
        """
        super().__init__(*args, **kwargs)
        if feat_size%patch_size!=0:
            raise ValueError('Patch size should devide feature size!')
        self.patch_resolution = feat_size//patch_size
        self.patch_size = patch_size
        self.output_nc = out_channels
        self.proj = nn.Conv2d(embed_dim, out_channels=out_channels * self.patch_size ** 2,
                              kernel_size=1, stride=1)
        self.merge = nn.PixelShuffle(patch_size)
        
    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        """
        (B, H*W/p^2, C') -> (B, C, H, W)
        """
        B, HW, C = x.shape
        x = x.transpose(1,2).view(B, C, self.patch_resolution, self.patch_resolution)
        x = self.merge(self.proj(x))

        return x

        


def window_partition(
        x: Tensor,
        window_size: int
) -> Tensor:
    """ (B, H, W C) -> (num_windows*B, window_size, window_size, C)
    
    Parameters:
        x (Tensor)              -- (B, H, W, C)
        window_size (int)       -- window size
    
    Return:
        windows (Tensor)        -- (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows




def window_reverse(
        windows: Tensor, 
        window_size: int, 
        H: int, 
        W: int
) -> Tensor:
    """ (num_windows*B, window_size, window_size, C) -> (B, H, W C)

    Reverse the effect of <window_partition>

    Parameters:
        windows (Tensor)        -- (num_windows*B, window_size, window_size, C)
        window_size (int)       -- size of the window
        H (int)                 -- height of image
        W (int)                 -- width of image
    """
    num_windows = H * W / (window_size ** 2)
    B = int(windows.shape[0]/num_windows)
    x = windows.view(B, H//window_size, window_size, W//window_size, window_size, -1)
    x = x.permute(0, 1, 2, 3, 4, 5).contiguous().view(B, H, W, -1)
    return x




def get_bias_index_table(
        window_size: int
) -> Tensor:
    """Get the pair_wise realtive position index for each token insidet the window.
    
    Parameter:
        window_size (int)       -- size of the window
    """
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    return relative_coords.sum(-1)




class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module.

    Attnetion = Softmax(Q*K.T/sqrt(dim) + B)*V
        Q = input*Wq;
        K = input*Wk;
        V = input*Wv;

    Attr:
        dim (int)                                           -- # of input channels (dimension)
        window_size (int)                                   -- size of the window
        num_heads (int)                                     -- # of attention heads
        scale (float | None, optional)                      -- scale to scale down QK.T
        relative_position_bias_table (nn.Parameter)         -- the learnable bias table
        qkv (nn.Linear)                                     -- the matrix to generate Q, K,and V
        atten_drop (nn.Dropout)                             -- dropout layer after Softmax
        proj (nn.Lineaar)                                   -- projection layer that does not change dimension
        proj_drop (nn.Dropout)                              -- the final dropout
        softmax (nn.Softmax)                                -- the Softmax layer
    """
    def __init__(
            self, 
            dim: int,                       
            window_size: int,                
            num_heads: int,                  
            qk_scale: Optional[float] = None,
            qkv_bias: Optional[bool] = True,    
            attn_drop: Optional[float] = 0.0,      
            proj_drop: Optional[float] = 0.0,      
            *args, **kwargs
    ) -> None:
        """
        Parameters:
            dim (int)                               -- # of input channels
            window_size (int)                       -- size of the window (we take height = width = size)
            num_heads (int)                         -- # of attention heads
            qkv_bias (bool, optional)               -- if set, add additional bias to q, k, and v matrices, `default: True`
            qk_scale (float | None, optional)       -- use this to scale QK.T instead of sqrt(d)
            attn_drop (float, optional)             -- dropout ratio of attention weight
            proj_drop (float, optional)             -- dropout ration of output
        """
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**0.5

        # bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # relative position index
        relative_position_index = get_bias_index_table(window_size)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(-1)

    def forward(
            self, 
            x: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tensor:
        """Attentiona calculation.
        
        Parameters:
            x (Tensor)                  -- input features with shape of (num_windows * B, N, C)
            mask (Tensor, optional)     -- mask with shape of (num_windows, window_size ** 2, window_size ** 2) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class MLP(nn.Module):
    """Multi-Layer Perceptrrons."""
    def __init__(
            self, 
            in_features: int,
            hidden_features: int,
            out_features: int,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0.0,
            *args, **kwargs
    ) -> None:
        """A 3 layer MLP.
        
        Parameters:
            in_features (int)       -- # of input neurons
            hidden_features (int)   -- # of neurons in the hidden layer
            out_features (int)      -- # of output neurons
            act_layer (nn.Module)   -- activation layer
            drop (float)            -- dropout rate
        """
        super().__init__(*args, **kwargs)
        self.fc_1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = act_layer()
        self.fc_2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(drop)

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        x = self.fc_1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc_2(x)
        x = self.drop(x)
        return x




class STB(nn.Module):
    """Modified Swin Transformer Block (STB) with DeepNorm settings.

    This part is modified base on the original swin-transformer:
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows` - https://arxiv.org/pdf/2103.14030

    ..notes:
        Added deepnorm, and allows both pre-norm and post-norm.
    """
    def __init__(
            self, 
            dim: int,
            input_size: int,
            num_heads: int,
            window_size: int = 6,
            shift_size: int = 0,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_scale: Optional[float] = None,
            attn_drop: float = 0.0,
            drop: float = 0.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            norm_before: bool = False,
            deepnorm: bool = True,
            n_layers: Optional[int] = None,
            *args, **kwargs
    ) -> None:
        """The Swim Transformer Block (STB).

        Parameters:
            dim (int)                           -- # of input channels
            input_size (int)                    -- size of the input feature
            num_heads (int)                     -- # of attention heads
            window_size (int)                   -- window size, `default: 6`
            shift_size (int)                    -- shift size for SW-MSA
            mlp_ratio (float)                   -- ratio of mlp hidden dim to embedding dim, `default: 4`
            qkv_bias (bool)                     -- set if add a bias to q, k ,v, `default: True`
            qk_scale (float, optional)          -- use this to scale qk instead of sqrt(d), `default: None`
            attn_drop (float)                   -- attention dropout rate, `default: 0.0`
            drop (float)                        -- dropout rate, `default: 0.0`
            act_layer (nn.Module, optional)     -- activation layer, `default: nn.GELU`
            norm_layer (nn.Module, optional)    -- normalization layer, `default: nn.LayerNorm`
            norm_before (bool)                  -- set if normalize before attn block or FFN
            deepnorm (bool)                     -- use deepnet residual branch
            n_layers (int)                      -- # of transformer layers in the whole network set this param if use deepnet
        """
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.input_size = input_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm_1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, qk_scale, qkv_bias, attn_drop, drop
        )
        
        self.norm_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop,
            act_layer=act_layer
        )

        if self.shift_size > 0:
            attn_mask = self.get_mask(self.input_size)
        else:
            attn_mask = None
        
        self.register_buffer('attn_mask', attn_mask)

        self.norm_before = norm_before

        if deepnorm:
            self.alpha = (2 * n_layers) ** 0.25
            self.beta = (8 * n_layers) ** (-0.25)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=self.beta)
        else:
            self.alpha = 1.0


    def residual_connection(
            self,
            x: Tensor,
            residual: Tensor
    ) -> Tensor:
        return x * self.alpha + residual


    def get_mask(
            self,
            feat_size: int
    ) -> Tensor:
        """Calculate the mask for SW-MSA.

        If this is a W-MSA block, then shift_size = 0, then the mask is all 0.

        Parameter:
            feat_size (int) -- the size of the input feature (H = W = feat_size)
        """
        mask = torch.zeros((1, feat_size, feat_size, 1))
        h_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                mask[:, h, w, :] = cnt
                cnt += 1

        
        mask_windows = window_partition(mask, self.window_size)  # (1, H, W, 1) -> (nW*1, Ws, Ws, 1)
        mask_windows = mask_windows.view(-1, self.window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask


    def forward(
            self,
            x: Tensor,
            x_size: int
    ) -> Tensor:
        """Foward for swin transformer block.

        (B, H*W, C) -> (B, H*W, C)
        
        Parameters:
            x (Tensor)      -- input tensor (B, H*W, C)
            x_size (int)    -- size of the input (H = W = size)
        """
        B, L, C = x.shape  # L = H*W

        residual = x


        if self.norm_before:
            x = self.norm_1(x)

        x = x.view(B, x_size, x_size, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,  -self.shift_size), dims=(1,2))
        else:
            shifted_x = x
        
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, Ws, Ws, C)
        x_windows = x_windows.view(-1, self.window_size ** 2, C)

        if self.input_size == x_size:
            attn_windows = self.attn(x_windows, mask = self.attn_mask)  # (nW*B, Ws*Ws, C)
        else:
            attn_windows = self.attn(x_windows, mask = self.get_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, x_size, x_size)  # (B, H, W, C)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        else:
            x = shifted_x
        
        x = x.view(B, x_size ** 2, C)  # (B, H*W, C)

        x = self.residual_connection(x, residual)

        if not self.norm_before:
            x = self.norm_1(x)


        # Feed-Forward Networks (MLP)
        residual = x
        if self.norm_before:
            x = self.norm_2(x)
        x = self.mlp(x)
        x = self.residual_connection(x, residual)
        if not self.norm_before:
            x = self.norm_2(x)

        return x  




class SEB(nn.Module):
    """Squeze-Excitation Block."""
    def __init__(
            self, 
            in_channels: int,
            hidden_factor: int,
            *args, **kwargs
    ) -> None:
        """Squeze-Excitation Block.
        
        Parameters:
            in_channels (int)   -- # of input channels
            hedden_factor (int) -- scale down the # of neurons by this factor in the hidden layer (r in the original paper)
        """
        super().__init__(*args, **kwargs)
        self.se_squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.se_excitation = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//hidden_factor), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channels//hidden_factor), in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        
        attn = self.se_squeeze(x)
        attn = self.se_excitation(attn)
        x = x*attn

        return x




class AHPF(nn.Module):
    """Adaptive High Pass Filter (AHPF)."""
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            padding
    ) -> None:
        """Initialize AHPF module.

        Parameters:
            in_channels (int)       -- # of input channels
            out_channels (int)      -- # of output channels
            kernel_size (int)       -- kernel size
            padding (int)           -- padding for the convolution layers
        """
        super().__init__()

        self.hp_factor = nn.Parameter(torch.ones(1)/10, requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1)/100, requires_grad=True)

        weight = torch.ones((in_channels, out_channels, kernel_size, kernel_size))
        self.padding = padding
        self.register_buffer('weight',weight)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        smooth_x = x - F.conv2d(x, self.weight * self.hp_factor, padding = self.padding) - self.beta
        return self.relu(smooth_x)





class AFEB(nn.Module):
    """Adaptive feature extraction block.
    """
    def __init__(
            self, 
            io_nc: int,
            growth_rate: int,
            n_conv_layers: int,
            k_size: int,
            *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.adaptive_high_pass = AHPF(in_channels=io_nc, out_channels=io_nc, kernel_size=3, padding=1)

        self.se = SEB(in_channels=io_nc, hidden_factor=4)

        self.rdb = blocks.ResidualDenseBlock(
            growth_rate0=io_nc, growth_rate=growth_rate,
            n_conv_layers=n_conv_layers, k_size=k_size
        )


    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        shortcut = x
        x = self.adaptive_high_pass(x) + shortcut
        x = self.se(x)
        x = self.rdb(x) + shortcut
        return x




class SAB(nn.Module):
    """Self-Attention Block."""
    def __init__(
            self, 
            io_nc: int,
            dim: int,
            patch_size: int,
            n_stb: int,
            input_size: int,
            num_heads: int,
            window_size: int = 4,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_scale: Optional[float] = None,
            attn_drop: float = 0.0,
            drop: float = 0.0,
            norm_layer: nn.Module = nn.LayerNorm,
            norm_before: bool = False,
            deepnorm: bool = True,
            n_layers: Optional[int] = None,
            *args, **kwargs
    ) -> None:
        """Combined-Attention Residual Block.

        Parameters:
            io_nc (int)                         -- # of input and output channels
            dim (int)                           -- # of embedded channles
            n_stb (int)                         -- # of STBs within this block
            patch_size (int)                    -- size of the patch
            input_size (int)                    -- size of the input feature
            num_heads (int)                     -- # of attention heads
            window_size (int)                   -- window size, `default: 6`
            mlp_ratio (float)                   -- ratio of mlp hidden dim to embedding dim, `default: 4`
            qkv_bias (bool)                     -- set if add a bias to q, k ,v, `default: True`
            qk_scale (float, optional)          -- use this to scale qk instead of sqrt(d), `default: None`
            attn_drop (float)                   -- attention dropout rate, `default: 0.0`
            drop (float)                        -- dropout rate, `default: 0.0`
            norm_layer (nn.Module, optional)    -- normalization layer, `default: nn.LayerNorm`
            norm_before (bool)                  -- set if normalize before attn block or FFN
            deepnorm (bool)                     -- use deepnet residual branch
            n_layers (int)                      -- # of transformer layers in the whole network set this param if use deepnet
        """
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            feat_size=input_size, patch_size=patch_size,
            in_channels=io_nc, embed_dim=dim 
        )

        self.patch_unembed = PatchUnembed(
            feat_size=input_size, patch_size=patch_size, 
            out_channels=io_nc, embed_dim=dim
        )

        self.transformers = nn.ModuleList([
            STB(dim=dim, input_size=input_size, num_heads=num_heads,
                shift_size=0 if (i%2==0) else int(window_size//2),
                window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, norm_layer=norm_layer, norm_before=norm_before, deepnorm=deepnorm,
                n_layers=n_layers) 
        for i in range(n_stb)])


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=io_nc, out_channels=io_nc, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=io_nc, out_channels=io_nc, kernel_size=3, padding=1),
            nn.ReLU()
        )



    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        global_identity = x
       

        # self-attention branch
        
        x_size = x.shape[2]//self.patch_size
        x = self.patch_embed(x)
        for transformer in self.transformers:
            x = transformer(x, x_size)
        x = self.patch_unembed(x)

        x = self.conv1(x)
        x = (self.conv2(x)* 0.1 + global_identity) 
        return x



class Upscale(nn.Module):
    def __init__(
            self, 
            input_nc: int,
            output_nc: int,
            sr_scale: int,
            *args, **kwargs
    ) -> None:
        """Finale upsampler block.
        
        Parameters:
            input_nc (int)      -- # of input channels
            output_nc (int)     -- # of output channels
            sr_scale (int)      -- the upscale factor
        """
        super().__init__(*args, **kwargs)
        self.upscale = nn.Sequential(
            blocks.Upsampler(
                blocks.default_conv, scale=sr_scale, ngf=input_nc
            ),
            nn.Conv2d(
                in_channels=input_nc, out_channels=output_nc, kernel_size=3, padding=1, stride=1
            )
        )
    
    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        return self.upscale(x)






class CASRModel(nn.Module):
    """Combined-Attention Residual Block.

    Attrs:
        depth (int)                         -- # of CARBs
        io_nc (int)                         -- # of input and output channels
        dim (int)                           -- # of embedded channles
        n_stb (int)                         -- # of STBs within one CARB
        patch_size (int)                    -- size of the patch
        input_size (int)                    -- size of the input feature
        num_heads (int)                     -- # of attention heads
        window_size (int)                   -- window size, `default: 6`
        mlp_ratio (float)                   -- ratio of mlp hidden dim to embedding dim, `default: 4`
        qkv_bias (bool)                     -- set if add a bias to q, k ,v, `default: True`
        qk_scale (float, optional)          -- use this to scale qk instead of sqrt(d), `default: None`
        attn_drop (float)                   -- attention dropout rate, `default: 0.0`
        drop (float)                        -- dropout rate, `default: 0.0`
        norm_layer (nn.Module, optional)    -- normalization layer, `default: nn.LayerNorm`
        norm_before (bool)                  -- set if normalize before attn block or FFN
        deepnorm (bool)                     -- use deepnet residual branch
        n_layers (int)                      -- # of transformer layers in the whole network set this param if use deepnet
    """
    def __init__(
            self, 
            opt: Namespace,
            *args, **kwargs
    ) -> None:
        """Initialize CASRModel
        
        Parameters:
            opt (Namespace)     -- options
        """
        super().__init__(*args, **kwargs)
        sr_scale = opt.scale
        depth = opt.depth
        input_nc = opt.input_nc

        io_nc = opt.growth_rate0

        n_stb = opt.n_stb
        dim = opt.embed_dim
        patch_size = opt.patch_size
        input_size = int(384//opt.scale)
        num_heads = opt.num_heads
        window_size = opt.window_size
        mlp_ratio = opt.mlp_ratio
        qkv_bias = opt.qkv_bias
        qk_scale = None
        attn_drop = opt.attn_drop
        drop = opt.drop
        output_nc = opt.output_nc
        norm_layer = nn.LayerNorm
        norm_before = opt.norm_before
        deepnorm = opt.deepnorm
        n_layers = n_stb
        growth_rate = opt.growth_rate
        n_conv_layers = opt.rdb_depth

        # shallow feature extraction
        self.sfe = nn.Sequential(
            nn.Conv2d(
                in_channels=input_nc, out_channels=io_nc, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU()
        )
        

        self.AFEs = nn.ModuleList([
            AFEB(
                io_nc=io_nc, growth_rate=growth_rate, n_conv_layers=n_conv_layers, k_size=3
            )
        for _ in range(depth)])

        self.GFF = nn.Sequential(
            nn.Conv2d(
                in_channels=depth * io_nc, 
                out_channels=io_nc,
                kernel_size=1, padding=0, stride=1
            ),
            nn.Conv2d(
                io_nc, io_nc,
                kernel_size=3, padding=1, stride=1
            )
        )

        self.SAB = SAB(
                io_nc=io_nc, dim=dim, patch_size=patch_size, n_stb=n_stb, input_size=input_size,
                num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, drop=drop, norm_layer=norm_layer,
                norm_before=norm_before, deepnorm=deepnorm, n_layers=n_layers
            )

        self.upscale = Upscale(input_nc=io_nc, output_nc=output_nc ,sr_scale=sr_scale)


        self.final_conv = nn.Sequential()
        
        

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        x = self.sfe(x)

        shortcut = x

        afeb_out = []
        for afeb in self.AFEs:
            x = afeb(x)
            afeb_out.append(x)

        x = self.GFF(torch.cat(afeb_out, dim=1))

        x = self.SAB(x)

        x = self.upscale(x + shortcut)
        return x