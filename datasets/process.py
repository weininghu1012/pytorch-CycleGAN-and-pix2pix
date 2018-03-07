# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from PIL import Image

#import Image

# functon that take input arguments as 'WB', 'Edge', etc to transform the images

def singleImageToBlackWhite(originalImagePath, outputImagePath):
    original_image = Image.open(originalImagePath)
    bw = original_image.convert('L')
    bw.save(outputImagePath)

def multipleImageToBlackWhite(fold_original, fold_output):
    # pass in the same directory when creating the processed images
    splits = os.listdir(fold_original)
    for sp in splits:
        img_fold_original = os.path.join(fold_original, sp)
        img_fold_output = os.path.join(fold_output, sp)

        if not os.path.isdir(img_fold_output):
            os.makedirs(img_fold_output)

        img_list = os.listdir(img_fold_original)
        num_imgs = len(img_list)
        print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
        for i in range(num_imgs):
            originalImageName = img_list[i]
            path_original = os.path.join(img_fold_original, originalImageName)
            outputImageName = originalImageName
            path_output = os.path.join(img_fold_output, outputImageName)

            # process
            singleImageToBlackWhite(path_original, path_output)
    print('All images finished')


if __name__ == "__main__":

    parser = argparse.ArgumentParser('create image pairs')
    parser.add_argument('--process_type', dest = 'process_type', help = 'The type you want the output image to tranfer into', type = str, default = 'WB')
    parser.add_argument('--fold_original', dest = 'fold_original', help = 'input directory for original', type = str)
    parser.add_argument('--fold_output', dest = 'fold_output', help = 'input directory for processed images', type = str)

    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg,  getattr(args, arg))

    # start to process the images
    if (args.process_type == 'BlackWhite'):
        multipleImageToBlackWhite(args.fold_original, args.fold_output)


