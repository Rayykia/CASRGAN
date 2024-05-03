"""Helper funcitions and classes for model construction.
"""

import torch.nn as nn
import functools
from torch import Tensor


class Identity(nn.Module):
    def forward(
            self, 
            x: Tensor
    ) -> Tensor:
        """Standard forward."""
        return x
    

def get_norm_layer(
        norm_type: str ='batch_norm'
) -> nn.Module:
    """Get a normalization layer.

    Parameters
        norm_type (str)     -- the normalization layer: [batch_norm | instance_norm | none]
    """
    if norm_type == "batch_norm":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    if norm_type == "instance_norm":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    if norm_type == "none":
        norm_layer = Identity()
    else:
        raise NameError(
            '{} is not found, please try batch_norm | instance_norm | none'.format(norm_type)
        )

    return norm_layer


# code test
if __name__=="__main__":
    nl = get_norm_layer('none')
    print(nl(3))