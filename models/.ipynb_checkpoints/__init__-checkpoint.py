"""This package contains modules related to model connstructions.

Author: Ruichen Deng

Example:
    >>> from models import create_model
    >>> model = creeate_model(opt)
"""

import torch
import importlib
import torch.nn as nn
import numpy as np

from collections import OrderedDict

from argparse import Namespace
from typing import Tuple




def find_model(model_name: str) -> nn.Module:
    """Import the wanted module. Return the model"""
    model_filename = "models."+model_name+"_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace("_","") + "Model"
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
    if model is None:
        print("{} model not found!".format(model_name))
        exit(0)
    return model


def create_model(opt:Namespace) -> nn.Module:
    """Create the model using the options.

    This is the main interfaace between `train.py / test.py`.
    
    Example:
        >>> from models import creat_model
        >>> model = create_model(opt)
    """
    model = find_model(opt.model.upper())
    if opt.model.lower() == 'srcnn':
        instance = model()
    else:
        instance = model(opt)
    
    return instance




def get_model_info(
        model: nn.Module, 
        input_size: Tuple[int, ...], 
        batch_size: int = -1, 
        device: str= 'cuda'
) -> None:
    """Display the model info.
    
    Information including: 
        -- # : parameters, trainable parameters, non-trainable parameters
        -- size : input, forward/backward pass, paramiters, the model

    Paremeters
    ----------
    model (nn.Module)
        -- the model
    input_size (Tuple)
        -- shape of the input tensor
    batch_size (int)
        -- batch size of the input tensor
    devive (str)
        -- [cuda | cpu]

    Example:
        >>> from models import create_model
        >>> from models import get_model_info
        >>> from options import TrainOptions
        >>> # read commandline
        >>> opt = TrainOptions().parse()
        >>> # create the model
        >>> model = create_model(opt)
        >>> # display the model information
        >>> get_model_info(model, opt.input_size, opt.batch_size)
    """

    model_info = OrderedDict()
    hooks = []
    n_conv_layers = OrderedDict()
    n_conv_layers['num'] = 0

    def _register_hook(module: nn.Module) -> None:
        def _hook(module, input, output):
            """Standard definition for a hook function."""
            layer_name = str(module.__class__).split('.')[-1]
            layer_idx = len(model_info)

            layer_key = '{}-{}'.format(layer_name, layer_idx+1)
            model_info[layer_key] = OrderedDict()
            model_info[layer_key]['input_shape'] = list(input[0].size())
            model_info[layer_key]['input_shape'][0] = batch_size

            if isinstance(output,(list, tuple)):
                model_info[layer_key]['output_shape'] = [[-1]+list(o.size())[1:] for o in output]
            else:
                model_info[layer_key]['output_shape'] = list(output.size())
                model_info[layer_key]['output_shape'][0] = batch_size

            n_params = 0
            n_conv = 0
            # gather # of weight parameters
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                n_params += torch.prod(torch.LongTensor(list(module.weight.size())))
                model_info[layer_key]['trainable'] = module.weight.requires_grad
                n_conv += 1
            # gather # of bias parameters
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                n_params += torch.prod(torch.LongTensor(list(module.bias.size())))
            model_info[layer_key]['n_params'] = n_params
            n_conv_layers['num'] += n_conv

        # list of hooks, recorded to remove these hooks later in order not 
        # to impact the model performance
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not module == model
        ):
            hooks.append(module.register_forward_hook(_hook))

    if device == 'cuda' and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    if isinstance(input_size, tuple):
        input_size = [input_size]

    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    model.apply(_register_hook)

    model(*x)

    for h in hooks:
        h.remove()

    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0

    for layer_key in model_info:
        total_params += model_info[layer_key]["n_params"]
        total_output += np.prod(model_info[layer_key]["output_shape"])
        if "trainable" in model_info[layer_key]:
            if model_info[layer_key]['trainable'] == True:
                trainable_params += model_info[layer_key]["n_params"]
    
    # calculate size: assume 4 byte per number
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024. ** 2))
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2))
    total_size = total_input_size + total_output_size + total_params_size

    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("Depth (# of conv layers): {0:,}".format(n_conv_layers['num']))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("================================================================")