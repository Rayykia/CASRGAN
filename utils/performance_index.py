"""Performance index used in evaluating the models.
Calculat PSNR and SSIM scores.

"""
import torch
import torch.nn.functional as F

from torch import Tensor




def PSNR(
        img1: Tensor, 
        img2: Tensor
) -> torch.float:
    """Calculate the PSNR between image 1 and image 2.
    
    Parameters:
        img1 (Tensor)       -- the generated image
        img2 (Tensor)       -- the reference image
    """
    with torch.no_grad():
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100.
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

 


def create_window(
        window_size, 
        sigma: float = 1.5,
        channel: int = 1,
        device: torch.device = 'cuda'
):
    """Create a gaussian kernel.
    
    Obtained by multiplying two matrices. (two one-dimensional gaussian distribution).
    Channel can be set to 3.
    
    Parameters:
        window_size (int)       -- the size of the gaussian kernel
        sigma (float)           -- variance of the gaussian kernel
        channel (int)           -- channels of the image
    """
    x = torch.arange(window_size, device=device).float()
    x -= window_size // 2
    gauss = torch.Tensor(torch.exp(-x**2/float(2*sigma**2)))
    gauss /= gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window




def SSIM(
        img1: Tensor, 
        img2: Tensor,
        window_size: int =11,
        sigma: float = 1.5,
        device: torch.device = 'cuda'
) -> float:
    """Calculate the SSIM between image 1 and image 2.
    
    Parameters:
        img1 (Tensor)           -- the generated image
        img2 (Tensor)           -- the reference image
        window_size (int)       -- the size of the gaussian kernel
        sigma (float)           -- variance of the gaussian kernel
    
    .. notes:
        Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y]
    """
    with torch.no_grad():

        window = create_window(window_size, sigma, device=device)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img1.size(1))

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return torch.mean(ssim_map).item()




def batch_PSNR(
        img_batch1: Tensor, 
        img_batch2: Tensor
) -> float:
    """Calculate the PSNR between batch 1 and batch 2.
    
    Parameters:
        img_batch1 (Tensor)         -- the generated batch
        img_batch2 (Tensor)         -- the reference batch
    """
    psnr = 0
    for img1, img2 in zip(img_batch1, img_batch2):
        psnr += PSNR(img1, img2)
    psnr /= len(img_batch1)
    return psnr.item()