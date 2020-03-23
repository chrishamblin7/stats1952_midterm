#get rank with respect to each class in a dataset, do this in a hacky single class way, because for some stupid reason your gpu memory is getting used up otherwise


from subprocess import call
import os
import argparse
import time

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_path", type = str, default = "/home/chris/projects/categorical_pruning/data/letters_and_enumeration/train")
	parser.add_argument("--model", type = str, default = '../models/custom_trained_enumandletters_seed10_0.934')
	parser.add_argument("--output_folder", type = str, default = "../rankings/lettersandenum_by_class/")
	parser.add_argument("--device", type = str, default = "0")
	args = parser.parse_args()
	#args.use_cuda = args.use_cuda and torch.cuda.is_available()

	return args



def make_dummy_dir(orig_path):
	path_split = orig_path.split('/')
	path_split[-1] = path_split[-1] + '_dummy'
	dummy_path = '/'.join(path_split)
	os.mkdir(dummy_path)
	subdirs = os.listdir(orig_path)
	for subdir in subdirs:
		os.mkdir(os.path.join(dummy_path,subdir))
	return dummy_path, subdirs


if __name__ == '__main__':
	args = get_args()

	if not os.path.exists(args.output_folder):
		os.mkdir(args.output_folder)

	dummy_path, labels = make_dummy_dir(args.train_path)


	for label in labels:
		print(label)

		call('CUDA_VISIBLE_DEVICES=%s python single_class.py --label %s --dummy_path %s --train_path %s --model %s --output_folder %s --no-crop'%(args.device,label,dummy_path,args.train_path,args.model,args.output_folder),shell=True )

		time.sleep(7)

	call('rm -r %s'%dummy_path,shell=True)

