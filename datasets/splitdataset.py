import os
import numpy as np
import random
import argparse
import shutil

def writeTrainTestVal(fold_output):
	if not os.path.isdir(fold_output):
		os.makedirs(fold_output)
	newPaths = []
	for path in ['train', 'test', 'val']:
		newPath = fold_output + '/' + path
		if not os.path.isdir(newPath):

			os.makedirs(newPath)
		newPaths.append(newPath)
	return newPaths


# Toso: this method definitely need refactoring!!!!
def allocateImages(newPaths, fold_original, train_split, test_split, val_split):


	img_list = os.listdir(fold_original)
	img_number = len(img_list)
	img_index = [i for i in range(img_number)]
	random.shuffle(img_index)
	train_index = img_index[:int(img_number*train_split)]
	test_index = img_index[int(img_number*train_split):int(img_number*(test_split + train_split))]
	val_index = img_index[int(img_number*(test_split + train_split)):]

	indexes = [train_index, test_index, val_index]
	for i in range(3):
		curIndex = indexes[i]


		for index in curIndex:

			img_name = img_list[index]
			path_original = os.path.join(fold_original, img_name)
			outputImageName = img_name
			path_output = os.path.join(newPaths[i], outputImageName)
			shutil.copy2(path_original, path_output)
	print('Finish')




if __name__ == "__main__":

    parser = argparse.ArgumentParser('create sub folders and image splits')
    parser.add_argument('--train_split', dest = 'train_split', help = 'The training split you wish to have', type = float, default = 0.8)
    parser.add_argument('--test_split', dest = 'test_split', help = 'The testing split you wish to have', type = float, default = 0.1)
    parser.add_argument('--val_split', dest = 'val_split', help = 'The val split you wish to have', type = float, default = 0.1)
    parser.add_argument('--fold_original', dest = 'fold_original', help = 'input directory for original', type = str)
    parser.add_argument('--fold_output', dest = 'fold_output', help = 'input directory for processed images', type = str)

    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg,  getattr(args, arg))

    newPaths = writeTrainTestVal(args.fold_output)
    allocateImages(newPaths, args.fold_original, args.train_split, args.test_split, args.val_split)



