import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from .performance_index import batch_PSNR, SSIM
import numpy as np

from argparse import Namespace
from torch import Tensor
from typing import Tuple


class VGG(nn.Module):
    """First 16 layers in the pre-trained VGG19."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        vgg = models.vgg19(True)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.features[:17]
        

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        return self.vgg(x)



class DownSalmpe(nn.Module):
    def __init__(self, input_channel, output_channel,  stride, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.down = nn.Sequential(
            DownSalmpe(64, 64, stride=2, padding=1),
            DownSalmpe(64, 128, stride=1, padding=1),
            DownSalmpe(128, 128, stride=2, padding=1),
            DownSalmpe(128, 256, stride=1, padding=1),
            DownSalmpe(256, 256, stride=2, padding=1),
            DownSalmpe(256, 512, stride=1, padding=1),
            DownSalmpe(512, 512, stride=2, padding=1),
        )
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.down(x)
        x = self.dense(x)
        return x




class GANModel():
    def __init__(
            self, 
            generator: nn.Module,
            opt: Namespace
    ) -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)
        self.discriminator = GANDiscriminator().to(self.device)
        self.vgg = VGG().to(self.device)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.batch_size = opt.batch_size

        self.lr = opt.lr

        self.g_optim = torch.optim.Adam(self.generator.parameters(),lr=self.lr)
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.g_scheduler = self._get_scheduler(self.g_optim, opt)
        self.d_scheduler = self._get_scheduler(self.d_optim, opt)

    @staticmethod
    def _get_scheduler(optimizer, opt):
        """Get learning rate scheduler."""
        if opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.7)
        elif opt.lr_policy == 'linear':
            scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=opt.lr_factor, total_iters=opt.lr_decay_iters)
        else:
            print('scheduler not set, currently only support [step | linear] scheduler')
            exit(0)
        return scheduler
        

    @torch.no_grad()
    def forward(
            self, 
            input: Tensor
    ) -> Tensor:
        self.generator.eval()
        return self.generator(input)
    

    def _g_loss(
            self, 
            recon_hr: Tensor, 
            real_hr: Tensor,
            x: Tensor,
            eta1: float = 5e-3,
            eta2: float = 1e-2
    ) -> Tensor:
        recon_hr = torch.cat([recon_hr for _ in range(3)],dim=1)
        real_hr = torch.cat([real_hr for _ in range(3)],dim=1)
        perceptual_loss = self.mse(
            self.vgg(recon_hr), self.vgg(real_hr)
        )

        gan_loss = torch.sum(-torch.log(x))

        l1_loss = self.l1(recon_hr, real_hr)

        return perceptual_loss + eta1 * gan_loss + eta2 * l1_loss
    

    def _d_loss(
            self,
            recon_score: Tensor,
            real_score: Tensor
    ) -> Tensor:
        recon_loss = self.bce(recon_score, torch.zeros_like(recon_score))
        real_loss = self.bce(real_score, torch.ones_like(real_score))
        return recon_loss + real_loss


    def _fit(self, trainloader, testloader) -> Tuple:
        print('training...')
        g_epoch_loss, d_epoch_loss = 0, 0
        epoch_psnr, epoch_ssim = 0, 0
        n_batch = len(trainloader)

        self.generator.train()
        self.discriminator.train()

        train_start_time = time.time()
        for lr_img, hr_img in tqdm(trainloader, ncols=64):
            lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)

            recon_hr = self.generator(lr_img)

            # ---------
            # generator
            # ---------
            recon_score = self.discriminator(recon_hr)
            g_loss = self._g_loss(recon_hr, hr_img, recon_score)

            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            # -------------
            # discriminator
            # -------------
            real_score = self.discriminator(hr_img)
            recon_score = self.discriminator(recon_hr.detach())
            d_loss = self._d_loss(recon_score, real_score)
            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            with torch.no_grad():
                g_epoch_loss += g_loss.item()
                g_epoch_loss /= self.batch_size
                d_epoch_loss += d_loss.item()
                d_epoch_loss /= self.batch_size
        train_time = time.time() - train_start_time
        epoch_psnr, epoch_ssim = self._eval(testloader)
        
        return g_epoch_loss, d_epoch_loss, epoch_psnr, epoch_ssim, train_time

    @torch.no_grad()
    def _eval(self, testloader):
        print('evaluating...')
        self.generator.eval()
        self.discriminator.eval()
        psnr, ssim = 0, 0
        num_iter = len(testloader)
        for lr_img, hr_img in testloader:
            lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
            recon_hr = self.generator(lr_img)

            psnr += batch_PSNR(recon_hr, hr_img)
            ssim += SSIM(recon_hr, hr_img)
        psnr /= num_iter
        ssim /= num_iter

        return psnr, ssim


    @torch.no_grad()
    def generate_and_save_images(
            self, epoch, test_input, opt
    ):
        """Save the process."""
        predictions = np.squeeze(self.forward(test_input).permute(0, 2, 3, 1).detach().cpu().numpy())
        plt.imsave('./imgs/process/{}GAN/x{}/epoch_{}.png'.format(opt.model.upper(), opt.scale, epoch), (predictions+1)/2)
        print('process saved to: ./imgs/process/{}GAN/x{}/epoch_{}.png'.format(opt.model.upper(), opt.scale, epoch))


    def optimize(self, opt, trainloader, testloader, test_seed):
        test_seed = test_seed.to(self.device)
        if opt.pre_train:
            self.generator.load_state_dict(torch.load(
                './checkpoints_/{}_{}_epoch{}.pt'.format(
                    opt.model.lower(), opt.scale, opt.pre_train_epoch
                )
            ))
            print('Parameters loaded succesfully from: ./checkpoints_/{}_{}_epoch{}.pt'.format(
                    opt.model.lower(), opt.scale, opt.pre_train_epoch
                ))
            
        for epoch in range(opt.n_epoches):

            g_epoch_loss, d_epoch_loss, epoch_psnr, epoch_ssim, epoch_train_time = self._fit(
                trainloader, testloader
            )

            print('epoch {} completed \t g loss: {:.3f} \t d loss: {:3f} \t psnr: {:.3f} dB \t ssim: {:.4f} \t time: {} min {} sec'.format(
                epoch, g_epoch_loss, d_epoch_loss, epoch_psnr, epoch_ssim, int(epoch_train_time//60), int(epoch_train_time%60)
            ))

            self.g_scheduler.step()
            self.d_scheduler.step()

            if (epoch+1) % opt.save_process_freq == 0:
                self.generate_and_save_images(epoch, test_seed, opt)
    
            if (epoch+1) % opt.checkpoint_freq == 0:
                torch.save(self.generator.state_dict(), "./checkpoints_/{}gan_g_{}_epoch{}.pt".format(
                    opt.model.lower(), opt.scale, epoch
                ))
                torch.save(self.discriminator.state_dict(), "./checkpoints_/{}gan_d_{}_epoch{}.pt".format(
                    opt.model.lower(), opt.scale, epoch
                ))
                print("epoch {} saved".format(epoch))
                
            # generate logs: loss
            print('recording loss... ', end='\t')
            with open('./logs/{}gan_{}_loss.txt'.format(
                opt.model.lower(), opt.scale
            ), mode='a') as f:
                f.write('epoch{:3d} \t g: {}; \t d: {}\n'.format(
                    epoch, g_epoch_loss, d_epoch_loss
                ))
            print('Done.')
            
            # generate logs: psnr
            print('recording psnr ...', end='\t')
            with open('./logs/{}gan_{}_psnr.txt'.format(
                opt.model.lower(), opt.scale
            ), mode='a') as f:
                f.write('epoch{:3d}:{}\n'.format(epoch, epoch_psnr))
            print('Done.')
            
            # generate logs: ssim
            print('recording ssim ...', end='\t')
            with open('./logs/{}gan_{}_ssim.txt'.format(
                opt.model.lower(), opt.scale
            ), mode='a') as f:
                f.write('epoch{:3d}:{}\n'.format(epoch, epoch_ssim))
            print('Done.')
            
            print()