# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
from PIL import Image
import PIL.Image

#import Image

# functon that take input arguments as 'WB', 'Edge', etc to transform the images
def singleImageToBlackWhite(originalImagePath, outputImagePath):
    original_image = Image.open(originalImagePath)
    bw = original_image.convert('L')
    bw.save(outputImagePath)

def multipleImageToProcess(fold_original, fold_output, process_type, dim = 64, do_crop = False):
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
            if (process_type == 'BW'):
                singleImageToBlackWhite(path_original, path_output)
            elif (process_type == 'RS'):
                singleImageToResize(path_original, path_output, dim, do_crop)

    print('All images finished')
    print('test')

def singleImageToResize(originalImagePath, outputImagePath, dim, do_crop):
    original_image = Image.open(originalImagePath)
    original_image = original_image.convert('RGB')
    out_shape = (dim, dim)
    if (do_crop):
        resize_shape = list(out_shape)
        if (original_image.width < original_image.height):
            resize_shape[1] = int(round(float(original_image.height) / original_image.width * dim))
        else:
            resize_shape[0] = int(round(float(original_image.width) / original_image.height * dim))
        # reference: https://stackoverflow.com/questions/23113163/antialias-vs-bicubic-in-pilpython-image-library
        original_image = original_image.resize(resize_shape, PIL.Image.BICUBIC)
        hw = int(original_image.width / 2)
        hh = int(original_image.height / 2)
        hd = int(dim/2)
        area = (hw - hd, hh - hd, hw + hd, hh + hd)
        output_image = original_image.crop(area)
    else:
        output_image = original_image.resize(out_shape, PIL.Image.BICUBIC)
    output_image.save(outputImagePath)




if __name__ == "__main__":

    parser = argparse.ArgumentParser('create image pairs')
    parser.add_argument('--fold_original', dest = 'fold_original', help = 'input directory for original', type = str)
    parser.add_argument('--fold_output', dest = 'fold_output', help = 'output directory for processed images', type = str)
    parser.add_argument('--process_type', dest = 'process_type', help = 'The type you want the output image to tranfer into', type = str, default = 'WB')
    

    # start to process the images
    parser.add_argument('--dim', dest = 'dim', help = 'The dimension your image wish to have', type = int, default = 64, required = False)
    parser.add_argument('--do_crop', dest = 'do_crop', help = 'If True, resizes shortest edge to target dimensions and crops other edge. If false, does non-uniform resize', type = bool, default = False, required = False)
    args = parser.parse_args()
    for arg in vars(args):
        print('[%s] = ' % arg,  getattr(args, arg))
    if (args.process_type == 'RS'):
        multipleImageToProcess(args.fold_original, args.fold_output, args.process_type, args.dim, args.do_crop)
    elif (args.process_type == 'BW'):
        multipleImageToProcess(args.fold_original, args.fold_output, args.process_type)
     

