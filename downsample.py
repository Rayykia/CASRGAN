"""Downsampling script for generating low-resolution images.

Example:
    Generate gray scale LR images with reduced size:
        >>> python downsample.py -g --hr_dir ./data/HR --lr_dir ./data/LR
    Generate gray scale LR images and use bicubic interpolation to upscale which to the high-resolution size: -k

    .. notes:
    Details of downsampling options can be found in downsample_options.py in package `options`
"""
import os
import cv2

from options.downsample_options import DownsampleOptions
from tqdm import tqdm


if __name__ == '__main__':
    opt = DownsampleOptions().parse()

    hr_dir = opt.hr_dir
    lr_dir = opt.lr_dir

    os.makedirs("{}".format(lr_dir), exist_ok=True)
    os.makedirs("{}/x2".format(lr_dir), exist_ok=True)
    os.makedirs("{}/x3".format(lr_dir), exist_ok=True)
    os.makedirs("{}/x4".format(lr_dir), exist_ok=True)


    # Downsample HR images
    hr_imgs = os.listdir(hr_dir)

    for filename in tqdm(hr_imgs):
    
        name, ext = os.path.splitext(filename)
    
        #Read HR image
        hr_img = cv2.imread(os.path.join(hr_dir, filename))
        hr_img_dims = (hr_img.shape[1], hr_img.shape[0])
    
        #Blur with Gaussian kernel of width sigma = 1
        if opt.gaussian_blur:
            hr_img = cv2.GaussianBlur(hr_img, (0,0), 1, 1)

        #Downsample image 2x
        lr_image_2x = cv2.resize(hr_img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        if opt.keepdims:
            lr_image_2x = cv2.resize(lr_image_2x, hr_img_dims, interpolation=cv2.INTER_CUBIC)

        if opt.gray_scale:
            lr_image_2x = cv2.cvtColor(lr_image_2x, cv2.COLOR_BGR2GRAY)

    
        cv2.imwrite(os.path.join(lr_dir + "/x2", filename.split('.')[0]+'x2'+ext), lr_image_2x)
    
        #Downsample image 3x
        lr_image_3x = cv2.resize(hr_img, (0,0), fx=(1 / 3), fy=(1 / 3), interpolation=cv2.INTER_AREA)

        if opt.keepdims:
            lr_image_3x = cv2.resize(lr_image_3x, hr_img_dims, interpolation=cv2.INTER_CUBIC)

        if opt.gray_scale:
            lr_image_3x = cv2.cvtColor(lr_image_3x, cv2.COLOR_BGR2GRAY)

    
        cv2.imwrite(os.path.join(lr_dir + "/x3", filename.split('.')[0]+'x3'+ext), lr_image_3x)
    
        # Downsample image 4x
        lr_img_4x = cv2.resize(hr_img, (0, 0), fx=0.25, fy=0.25,
                               interpolation=cv2.INTER_AREA)
        if opt.keepdims:
            lr_img_4x = cv2.resize(lr_img_4x, hr_img_dims,
                                   interpolation=cv2.INTER_CUBIC)
        if opt.gray_scale:
            lr_img_4x = cv2.cvtColor(lr_img_4x, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(os.path.join(lr_dir + "/x4", filename.split('.')[0]+'x4'+ext), lr_img_4x)