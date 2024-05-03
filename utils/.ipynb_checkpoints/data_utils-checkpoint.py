import torch.utils.data as data
import os

from torchvision import transforms
from PIL import Image

from argparse import Namespace
from typing import Any





class DataPool(data.Dataset):
    def __init__(self, hr_paths, lr_paths) -> None:
        super().__init__()
        self.HR_paths = hr_paths
        self.LR_paths = lr_paths

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index) -> Any:
        LR_pil_img = Image.open(self.LR_paths[index])
        LR_tensor_img = self.transform(LR_pil_img)

        HR_pil_img = Image.open(self.HR_paths[index])
        HR_tensor_img = self.transform(HR_pil_img)

        return LR_tensor_img, HR_tensor_img
    
    def __len__(self) -> int:
        return len(self.HR_paths)




def create_dataloader(opt: Namespace):
    """Create a datloader given the options.
    
    This functions is the main interface between the data and `train.py | test.py`

    Example:
        >>> from data_utils import create_dataloader
        >>> dataloader = create_dataloader(opt)
    """
    scale = opt.scale

    train_hr_dir = opt.hr_dir
    train_lr_dir = opt.lr_dir
    hr_imgs = os.listdir(train_hr_dir)
    hr_paths = [train_hr_dir + "/" + x for x in hr_imgs]
    lr_imgs = [x.split(".")[0]+'x{}'.format(scale) for x in hr_imgs]
    lr_paths = [train_lr_dir + '/x{}/{}.png'.format(scale, x) for x in lr_imgs]

    val_hr_dir = opt.eval_hr_dir
    val_hr_imgs = os.listdir(val_hr_dir)
    val_hr_paths = [val_hr_dir + '/' + x for x in val_hr_imgs]
    val_lr_imgs = [x.split(".")[0]+'x{}'.format(scale) for x in val_hr_imgs]
    val_lr_paths = [opt.eval_lr_dir + '/x{}/{}.png'.format(scale, x) for x in val_lr_imgs]

    train_loader = data.DataLoader(
        DataPool(hr_paths, lr_paths), batch_size=opt.batch_size, num_workers=opt.num_threads
    )
    val_loader = data.DataLoader(
        DataPool(val_hr_paths, val_lr_paths), batch_size=opt.batch_size, num_workers=opt.num_threads
    )
    return train_loader, val_loader