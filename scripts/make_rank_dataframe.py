#make a pandas dataframe object of rankings
import os
import torch
import pandas as pd


'''
file_dict = {'animate':torch.load('model_animate_imgnet_6_epochs_rank_v2.pt'),
			 'inanimate':torch.load('model_inanimate_imgnet_6_epochs_rank_v2.pt'),
			 'letters':torch.load('model_letters_imgnet_6_epochs_rank_v2.pt'),
			 'faces':torch.load('model_faces_50adam_lr.001_rank.pt'),
			  'enumeration':torch.load('model_enumeration_rank.pt')}
'''
labels = os.listdir('/home/chris/projects/categorical_pruning/data/letters_and_enumeration/train')
seeds = ['1','2','3','4','5','6','7','8','9','10']


biglist = []

for seed in seeds:
	for label in labels:
		ranks = torch.load('lettersandenum_by_class/%s_seed%s_rank.pt'%(label,seed))
		for i in range(len(ranks)):
			biglist.append([ranks[i][0],ranks[i][1],ranks[i][2],ranks[i][3],label,seed])

column_names = ['filter_num','layer','filter_num_by_layer','rank_activation','class','model']

df = pd.DataFrame(biglist,columns=column_names)
df.to_feather('letterandenum_multimodel_ranks.feather')
