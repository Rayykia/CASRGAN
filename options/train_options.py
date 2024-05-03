from .base_options import BaseOptions

from argparse import ArgumentParser


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.

    options:
        val_dir VAL_DIR         -- the directory of validation images (default: ./data/val)
        batch_size              -- the input batch size (default: 16)
        scale                   -- scale of the SR algorithm: [2 | 4] (default: None)
        save_process_freq       -- frequency of saving the latest results (default: 3)
        checkpoint_freq         -- frequency of saving checkpoints at the end of epochs (default: 30)
        continue_train, c       -- continue training: load the latest model (default: False)
        load_epoch              -- the starting epoch count (default: 0)
        loss_fn                 -- name of the loss function (default: l1)
        n_epoches               -- # of epochs with the initial learning rate (default: 60)
        lr LR                   -- initial learning rate for adam (default: 0.0001)
        lr_policy               -- learning rate policy. [step | linear] (default: linear)
        lr_decay_iters          -- total decay iterations for learning rate (default: 30)
        lr_factor               -- lr factor for setting lr scheduler (default: 0.5)
        pre_train               -- load pre-trained paramaters for GAN training (default: False)
        pre_train_epoch         -- load this epoch as GAN pre-trained model (defaule: 59)
    """

    def __init__(self) -> None:
        super().__init__()

    def initialize_options(self, parser: ArgumentParser):
        parser = BaseOptions.initialize_options(self, parser)

        parser.add_argument('--val_dir', type=str, default='./data/val', help='the directory of validation images')
        # dataset
        parser.add_argument('--batch_size', type=int, default=16, help='the input batch size')
        # network saving and loading parameters
        parser.add_argument('--scale', type=int, help='scale of the SR algorithm: [2 | 4]')
        parser.add_argument('--save_process_freq', type=int, default=3, help='frequency of saving the latest results')
        parser.add_argument('--checkpoint_freq', type=int, default=30, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('-c', '--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--load_epoch', type=int, default=0, help='the starting epoch count')
        
        # training parameters
        parser.add_argument('--loss_fn', type=str, default='l1', help='name of the loss function')
        parser.add_argument('--n_epoches', type=int, default=60, help='# of epochs with the initial learning rate')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [step | linear]')
        parser.add_argument('--lr_decay_iters', type=int, default=30, help='total decay iterations for learning rate')
        parser.add_argument('--lr_factor', type=float, default=0.5, help='lr factor for setting lr scheduler')

        # GAN training
        parser.add_argument('--pre_train', action='store_true', help = 'load pre-trained paramaters for GAN training')
        parser.add_argument('--pre_train_epoch', type=int, default=59, help='load this epoch as GAN pre-trained model')
        
        self.isTrain = True
        return parser

if __name__ == '__main__':
    opt = TrainOptions().parse()