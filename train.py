"""Train the models.

Example:
    train <model> with super-resolution <scale>
        >>> python train.py --model <model> --scale <scale>
    for SRCNN and DRRN, bicbiced LR image directory <lr_dir>  and bicubiced eval LR 
    directory <--eval_lr_dir> requires to be specified by adding: 
        --lr_dir <lr_dir> --eval_lr_dir <eval_lr_dir>

    train a GAN with <model> as generator with super-resolution <scale>
        >>> python train.py --model <model> --gan

    .. notes:
    More options can be found in base_options.py and train_options.py in package `options`
"""
import torch
import time

from torchvision.transforms import ToTensor
from PIL import Image

from options.train_options import TrainOptions
from utils.data_utils import create_dataloader
from models import create_model
from utils.train_utils import create_directories
from utils.train_utils import get_scheduler
from utils.train_utils import train_CNN
from utils.gan_utils import GANModel

import warnings
warnings.filterwarnings("ignore")


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    param_size = (param_size) / 1024 / 1024
    return param_sum, param_size



if __name__ == '__main__':
    create_directories()
    opt = TrainOptions().parse()
    train_loader, val_loader = create_dataloader(opt)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    print('Training Set Size: {} \t  Validation Set Size: {}'.format(train_size, val_size))

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('================================================================')

    if not opt.gan:  # if the model is not a GAN
        model = create_model(opt)
        print("{} model created successfully.".format(opt.model.upper()))
        model.to(device)
        param_sum, param_size = getModelSize(model)
        print('Total params: {}'.format(param_sum))
        print('Param size (MB): {:.3f}'.format(param_size))
        # print(model)
        print('================================================================')
        
    
        
        if opt.model.lower() in ['edsr', 'drrn', 'rdn', 'casr']:
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        elif opt.model.lower() == 'srcnn':
            optimizer = torch.optim.Adam([
                    {'params': model.patch_extraction.parameters()},
                    {'params': model.non_linear_mapping.parameters()},
                    {'params': model.reconstruction.parameters(), 'lr': opt.lr * 0.1}
                ], lr=opt.lr)
        else:
            print('Optimizer not set.')
            exit(0)
    
        scheduler = get_scheduler(optimizer, opt)
    
        if opt.continue_train is True:
            model.load_state_dict(torch.load("./checkpoints_/{}_{}_epoch{}.pt".format(
                opt.model.lower(), opt.scale, opt.load_epoch
            )))
    
    
    
        if opt.model.lower() in ['srcnn', ' drrn']:
            test_path = "./data/process_seed/0_0_2x{}_b.png".format(opt.scale)
        else:
            test_path = "./data/process_seed/0_0_2x{}.png".format(opt.scale)
    
    
        test_tensor = ToTensor()(Image.open(test_path).convert("L"))
        test_tensor = test_tensor.unsqueeze(dim=1)
        test_tensor = test_tensor.to(device)
    
    
    
        # CNN based
        start_time = time.time()
        train_CNN(
            opt, model, train_loader, val_loader, optimizer, scheduler, device, 
            test_input=test_tensor
        )  # training process
        total_time = (time.time() - start_time)//1
    
        print('End of epoch {} from epoch {} \t Total Time: {}h {} min {} sec'.format(
            opt.n_epoches+opt.epoch_count, opt.epoch_count, int(total_time//3600), int(total_time%3600//60), int(total_time%3600%60)
        ))

    else:  # use the given model as the generator of the GAN and train the GAN
        gan = GANModel(create_model(opt), opt)
        print('Generative Adversarial Network with {} created successfully.'.format(opt.model.upper()))
        gen_param_num, gen_param_size = getModelSize(gan.generator)
        dis_param_num, dis_param_size = getModelSize(gan.discriminator)
        total_param = gen_param_num + dis_param_num
        total_size = gen_param_size + dis_param_size
        print('Generator params: {}'.format(gen_param_num))
        print('Generator size (MB): {:.3f}'.format(gen_param_size))
        print('Discriminator params: {}'.format(dis_param_num))
        print('Discriminator size (MB): {:.3f}'.format(dis_param_size))
        print('-------------------------------')
        print('Total params: {}'.format(total_param))
        print('Totsl size (MB): {:.3f}'.format(total_size))
        print('================================================================')


        if opt.model.lower() in ['srcnn', ' drrn']:
            test_path = "./data/process_seed/0_0_2x{}_b.png".format(opt.scale)
        else:
            test_path = "./data/process_seed/0_0_2x{}.png".format(opt.scale)
    
    
        test_tensor = ToTensor()(Image.open(test_path).convert("L"))
        test_tensor = test_tensor.unsqueeze(dim=1)
        test_tensor = test_tensor.to(device)

        start_time = time.time()
        gan.optimize(opt, trainloader=train_loader, testloader=val_loader, test_seed=test_tensor)  # training process
        total_time = (time.time() - start_time)//1

        print('End of epoch {} from epoch {} \t Total Time: {}h {} min {} sec'.format(
            opt.n_epoches+opt.epoch_count, opt.epoch_count, int(total_time//3600), int(total_time%3600//60), int(total_time%3600%60)
        ))