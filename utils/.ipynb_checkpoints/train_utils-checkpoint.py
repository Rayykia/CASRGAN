import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import time
import copy

from torch.optim import lr_scheduler
from .performance_index import batch_PSNR, SSIM

from torch.utils.data import DataLoader
from typing import Callable, Tuple
from torch import Tensor
from argparse import Namespace
from tqdm import tqdm


def create_directories():
    """Create directories for the traning and evaluation process.

    --- [checkpoints_ | logs] --- [SRCNN | EDSR | DRRN | RDN | CASR] --- [x2 | x3 | x4]
    --- imgs --- [process | eval] --- [SRCNN | EDSR | DRRN | RDN | CASR] --- [x2 | x3 | x4]
    """
    dir_list = ['checkpoints_', 'logs', 'imgs']
    imgs_sub_dir_list = ['process', 'eval']
    model_list = ['SRCNN', 'EDSR', 'DRRN', 'RDN', 'CASR', 'EDSRGAN', 'RDNGAN', 'CASRGAN']
    scale_list = ['x2', 'x3', 'x4']

    for dir in dir_list:
        os.makedirs('./{}'.format(dir), exist_ok=True)
    for dir in imgs_sub_dir_list:
        for model in model_list:
            for scale in scale_list:
                os.makedirs('./imgs/{}/{}/{}'.format(dir, model, scale), exist_ok=True)




def get_loss_fn(opt):
    """Get loss function: [L1 | L2]
    
    .. notes: 
        Requires to be update if other loss functions are to be used.
    """
    if opt.loss_fn.lower() == 'l1':
        loss_fn = nn.L1Loss()
    elif opt.loss_fn.lower() in ['l2', 'mse']:
        loss_fn = nn.MSELoss()
    else:
        print('loss function not set, currently only support l1 and l2 loss')
        exit(0)
    return loss_fn




def get_scheduler(optimizer, opt: Namespace):
    """Get learning rate scheduler."""
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.7)
    elif opt.lr_policy == 'linear':
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=opt.lr_factor, total_iters=opt.lr_decay_iters)
    else:
        print('scheduler not set, currently only support [step | linear] scheduler')
        exit(0)
    return scheduler




def fit_CNN(
        model: nn.Module, 
        loss_fn: Callable[..., Tensor],
        optimizer: torch.optim.Optimizer, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        device: torch.device,
        batch_size: int
) -> Tuple:
    """Train the model for 1 epoch.
    
    Returns
    -------
    epoch_train_time
        -- training time
    epoch_train_loss
        -- training loss
    epoch_psnr
        -- PSNR on validation set
    epoch_ssim
        -- SSIM on validation set
    """
    epoch_start_time = time.time()
    epoch_loss, epoch_psnr, epoch_ssim = 0, 0, 0

    model.train()
    for lr_img, hr_img in tqdm(train_loader, ncols=64):
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        recon_hr = model(lr_img)
        loss = loss_fn(recon_hr, hr_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        with torch.no_grad():
            epoch_loss += loss.item()
    epoch_train_time = (time.time() - epoch_start_time)
    model.eval()
    with torch.no_grad():
        num_iter = len(val_loader)
        for lr_img, hr_img in val_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            recon_hr = model(lr_img)
            epoch_psnr += batch_PSNR(recon_hr, hr_img)
            epoch_ssim += SSIM(recon_hr, hr_img)
        epoch_psnr /= num_iter
        epoch_ssim /= num_iter
    return epoch_train_time, epoch_loss, epoch_psnr, epoch_ssim




def generate_and_save_images(model, epoch, test_input, opt):
    """Save the process."""
    predictions = np.squeeze(model(test_input).permute(0, 2, 3, 1).detach().cpu().numpy())
    plt.imsave('./imgs/process/{}/x{}/epoch_{}.png'.format(opt.model.upper(), opt.scale, epoch), (predictions + 1)/2)




def train_CNN(
        opt: Namespace,
        model: nn.Module, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler.LRScheduler, 
        device: torch.device, 
        test_input: Tensor,
):
    """Train the CNN-based network.
    
    Interface to train.py

    Parameters
    ----------
        -- opt (Namespace)
            training options
        -- model (nn.Module)
            model to be trained
        -- train_loader (DataLoader)
            dataloder of the training dataset
        -- val_loader (DataLoader)
            dataloader of the validation dataset
        -- optimizer (Optimizer)
            the optimizer of the model
        -- scheduler (Scheduler)
            the learning rate scheduler
        -- device (torch.duvice)
            the device to operate on
        -- test_input (Tensor)
            test tensor used to generate the process images
    """
    test_input = test_input.to(device)

    loss_fn = get_loss_fn(opt)

    min_loss = 1e9
    
    for epoch in range(opt.n_epoches):
        # epoches already trained
        if opt.continue_train:
            epoch = epoch + opt.load_epoch + 1
        print('epoch {} begins'.format(epoch))
        epoch_train_time, epoch_loss, epoch_psnr, epoch_ssim  = fit_CNN(
            model, loss_fn, optimizer, train_loader, val_loader, device, opt.batch_size
        )
        
        print('epoch {} completed \t loss: {:.3f} \t psnr: {:.3f} dB \t ssim: {:.4f} \t time: {} min {} sec'.format(
            epoch, epoch_loss, epoch_psnr, epoch_ssim, int(epoch_train_time//60), int(epoch_train_time%60)
        ))

        scheduler.step()
        
        if epoch_loss <= min_loss:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            min_loss = epoch_loss
            
        if (epoch+1) % opt.save_process_freq == 0:
            generate_and_save_images(model, epoch, test_input, opt)

        if (epoch+1) % opt.checkpoint_freq == 0:
            torch.save(best_state_dict, "./checkpoints_/{}_{}_epoch{}.pt".format(
                opt.model.lower(), opt.scale, best_epoch
            ))
            print("epoch {} saved".format(epoch))
            
        print()
        # generate logs: loss
        with open('./logs/{}_{}_loss.txt'.format(
            opt.model.lower(), opt.scale
        ), mode='a') as f:
            f.write('epoch{:3d}:{}\n'.format(epoch, epoch_loss))

        # generate logs: psnr
        with open('./logs/{}_{}_psnr.txt'.format(
            opt.model.lower(), opt.scale
        ), mode='a') as f:
            f.write('epoch{:3d}:{}\n'.format(epoch, epoch_psnr))

        # generate logs: ssim
        with open('./logs/{}_{}_ssim.txt'.format(
            opt.model.lower(), opt.scale
        ), mode='a') as f:
            f.write('epoch{:3d}:{}\n'.format(epoch, epoch_ssim))
        
