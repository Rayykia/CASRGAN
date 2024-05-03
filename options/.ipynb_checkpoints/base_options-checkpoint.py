import argparse

from argparse import ArgumentParser

class BaseOptions():
    """Options used in both training and testing.

    options:
        -h, --help          -- show this help message and exit
        hr_dir              -- the directory of high resolution images (default: ./data/HR)
        lr_dir              -- the directory of the low resolution images (default: ./data/LR)
        eval_hr_dir         -- the directory of the high resolution evaluation images (default: ./data/val/HR)
        eval_lr_dir         -- the directory of the low resolution evaluation images (default: ./data/val/LR)
        device              -- compute device: [cuda | cpu] (default: cuda)
        checkpoint_dir      -- model checkpoints are saved in this directory (default: ./checkpoints_)
        process_dir         -- images generated during the training process are saved here (default: None)
        num_threads         -- # threads for loading data (default: 16)
        model               -- choose the model to use [srcnn | srgan] (default: None)
        gan                 -- train a GAN with `model` as generator (defaulr: False)
        input_nc            -- # of input channels: 1 for gray scale 3 for RGB (default: 1)
        output_nc           -- # of output channels: 1 for gray scale 3 for RGB (default: 1)
        act                 -- activation function: [relu | prelu] (default: r   
        n_resblocks         -- EDSR_PARAM: # of residule blockes (default: 32)
        n_feats             -- EDSR_PARAM: # of feature channels in the residule blocks (default:    
        growth_rate0        -- RDN_PARAM: G0 (default: 64)
        growth_rate         -- RDN_PARAM: G (default: 32)
        rdb_depth           -- RDN_PARAM: C, # of conv layers to use within one RDB (default: 6)
        rdn_depth           -- RDN_PARAM: D, # of RDBs to use (default: 20)
        rdn_ksize           -- RDN_PARAM: kernel size (default   
        drrn_rb_depth       -- DRRN_PARAM: B, # of RBs (default: 1)
        drrn_ru_depth       -- DRRN_PARAM: U, # of RU within one RB (default: 25)
        drrn_ksize          -- DRRN_PARAM: kernel size (default: 3)
        drrn_ngf            -- DRRN_PARAM: # of filters (default:    
        depth               -- CASRGAN_PARAM: # of CARBs (default: 16)
        io_nc               -- CASRGAN_PARAM: # of input and output channels (default: 256)
        n_stb               -- CASRGAN_PARAM: # of STBs within one CARB (default: 2)
        embed_dim           -- CASRGAN_PARAM: # of embedded channles (default: 96)
        patch_size          -- CASRGAN_PARAM: size of the patch (default: 6)
        num_heads           -- CASRGAN_PARAM: # of attention heads (default: 6)
        window_size         -- CASRGAN_PARAM: window size, `default: 4` (default: 4)
        mlp_ratio           -- CASRGAN_PARAM: ratio of mlp hidden dim to embedding dim, `default: 4` (default: 4.0)
        qkv_bias            -- CASRGAN_PARAM: set if add a bias to q, k ,v, `default: True` (default: True)
        attn_drop           -- CASRGAN_PARAM: attention dropout rate, `default: 0.0` (default: 0.0)
        drop DROP           -- CASRGAN_PARAM: dropout rate, `default: 0.0` (default: 0.0)
        norm_before         -- CASRGAN_PARAM: norm before attn and FFN (default: False)
        deepnorm            -- CASRGAN_PARAM: use DeepNorm (default: False)
    """

    def __init__(self) -> None:
        self.initialized=False
        self.isTrain = None

    def initialize_options(self, parser: ArgumentParser):
        """This method initialize the arguments in the parser."""
        
        parser.add_argument('--hr_dir', type=str, default='./data/HR', help='the directory of high resolution images')
        parser.add_argument('--lr_dir', type=str, default='./data/LR', help='the directory of the low resolution images')
        parser.add_argument('--eval_hr_dir', type=str, default='./data/val/HR', help='the directory of the high resolution evaluation images')
        parser.add_argument('--eval_lr_dir', type=str, default='./data/val/LR', help='the directory of the low resolution evaluation images')
        parser.add_argument('--device', type=str, default='cuda', help='compute device: [cuda | cpu]')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_', help='model checkpoints are saved in this directory')
        parser.add_argument('--process_dir', type=str, help='images generated during the training process are saved here')

        parser.add_argument('--num_threads', default=16, type=int, help='# threads for loading data')

        # model
        parser.add_argument('--model', type=str, help='choose the model to use [srcnn | srgan]')
        parser.add_argument('--gan', action='store_true', help = 'train a GAN with `model` as generator')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input channels: 1 for gray scale 3 for RGB')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output channels: 1 for gray scale 3 for RGB')
        parser.add_argument('--act', type=str, default='relu', help='activation function: [relu | prelu]')

        # EDSR parameters
        parser.add_argument('--n_resblocks', type=int, default=32, help='EDSR_PARAM: # of residule blockes')
        parser.add_argument('--n_feats', type=int, default=256, help='EDSR_PARAM: # of feature channels in the residule blocks') 

        # DRRN parameters
        parser.add_argument('--drrn_rb_depth', type=int, default=1, help='DRRN_PARAM: B, # of RBs')
        parser.add_argument('--drrn_ru_depth', type=int, default=25, help='DRRN_PARAM: U, # of RU within one RB')
        parser.add_argument('--drrn_ksize', type=int, default=3, help='DRRN_PARAM: kernel size')
        parser.add_argument('--drrn_ngf', type=int, default=128, help='DRRN_PARAM: # of filters')

        # RDN parameters
        parser.add_argument('--growth_rate0', type=int, default=64, help='RDN_PARAM: G0')
        parser.add_argument('--growth_rate',type=int, default=32, help='RDN_PARAM: G')
        parser.add_argument('--rdb_depth', type=int, default=6, help='RDN_PARAM: C, # of conv layers to use within one RDB')
        parser.add_argument('--rdn_depth', type=int, default=20, help='RDN_PARAM: D, # of RDBs to use')
        parser.add_argument('--rdn_ksize', type=int, default=3, help='RDN_PARAM: kernel size')

        # CASRGAN parameters
        parser.add_argument('--depth', type=int, default=20, help='CASR_PARAM: # of CARBs')
        parser.add_argument('--io_nc', type=int, default=128, help='CASR_PARAM: # of input and output channels')
        parser.add_argument('--n_stb', type=int, default=2, help='CASR_PARAM: # of STBs within one CARB')
        parser.add_argument('--embed_dim', type=int, default=180, help='CASR_PARAM: # of embedded channles')
        parser.add_argument('--patch_size', type=int, default=6,  help='CASR_PARAM: size of the patch')
        parser.add_argument('--num_heads', type=int, default=6, help='CASR_PARAM: # of attention heads')
        parser.add_argument('--window_size', type=int, default=4, help='CASR_PARAM: window size, `default: 4`')
        parser.add_argument('--mlp_ratio', type=float, default=4.0,help='CASR_PARAM: ratio of mlp hidden dim to embedding dim, `default: 4`')
        parser.add_argument('--qkv_bias', action='store_false', help='CASR_PARAM: set if add a bias to q, k ,v, `default: True`')
        parser.add_argument('--attn_drop', type=float, default=0.0, help='CASR_PARAM: attention dropout rate, `default: 0.0`')
        parser.add_argument('--drop', type=float, default=0.0, help='CASR_PARAM: dropout rate, `default: 0.0`')
        parser.add_argument('--norm_before', action='store_true', help = 'CASR_PARAM: norm before attn and FFN')
        parser.add_argument('--deepnorm', action='store_true', help = 'CASR_PARAM: use DeepNorm')


        self.initialized = True
        return parser
    
    def read_commandline(self, args =  None):
        """Read the commandline and initialize the parser with absic opotions.
        
        If the parser is already initialized, skip initialize_options().
        Add model and dataset options.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize_options(parser)

        self.parser = parser
        return parser.parse_args(args)
    
    def parse(self, args = None):
        opt = self.read_commandline(args)
        opt.isTrain = self.isTrain

        self.opt = opt
        return self.opt
    
if __name__ == "__main__":
    opt = BaseOptions().parse()