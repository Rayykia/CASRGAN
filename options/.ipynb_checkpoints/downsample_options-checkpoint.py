import argparse
from argparse import ArgumentParser

class DownsampleOptions():
    """Options used in gatering low resolution images from high resolution images.
    
    options:
      -h, --help            show this help message and exit
      --hr_dir              source, the directory of high resolution images (default: ./data/HR)
      --lr_dir              destination, the directory of the low resolution images (default: ./data/LR)
      -k, --keepdims        keep original image dimensions in downsampled images (default: False)
      -g, --gray_scale      downsample the images to gray scale iamges (default: False)
      --gaussian_blur       use gaussian kernel to blur the images while downsampeling
    """
    def __init__(self) -> None:
        self.initialized = False


    def initialize_options(self, parser: ArgumentParser):
        parser.add_argument('--hr_dir', default='./data/HR', help='the directory of high resolution images')
        parser.add_argument('--lr_dir', type=str, default='./data/LR', help='the directory of the low resolution images')
        parser.add_argument('-k', '--keepdims', action='store_true', help='keep original image dimensions in downsampled images')
        parser.add_argument('-g', '--gray_scale', action='store_true', help='downsample the images to gray scale iamges')
        parser.add_argument('--gaussian_blur', action='store_true', help='use gaussian kernel to blur the images while downsampeling')
        self.initialized = True
        return parser
    

    def read_commandline(self):
        """Read the commandline and initialize the parser with absic opotions.
        
        If the parser is already initialized, skip initialize_options().
        Add model and dataset options.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize_options(parser)

        self.parser = parser
        return parser.parse_args()
    

    def parse(self):
        opt = self.read_commandline()
        self.opt = opt
        return self.opt