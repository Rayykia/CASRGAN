"""Get information about the model.

Example:
    Gather information about <model> with <scale>, <batch_size>
        >>> python get_info.py --model <model> --scale <scale> --batch_size <batch_size>
"""

import torch
from options.train_options import TrainOptions
from utils.data_utils import create_dataloader
from models import create_model
from models import get_model_info



if __name__ == '__main__':
    opt = TrainOptions().parse()
    train_loader, val_loader = create_dataloader(opt)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    print('Training Set Size: {} \t  Validation Set Size: {}'.format(train_size, val_size))

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    model = create_model(opt)
    model.to(device)
    if opt.model.lower() in ['srcnn','drrn']:
        input_size = (opt.input_nc, 384, 384)
    else:
        input_size = (opt.input_nc, int(384/opt.scale), int(384/opt.scale))
    get_model_info(model, input_size, batch_size=opt.batch_size)