"""
Filename: split_folds.py

Authors:
	Jason Youn -jyoun@ucdavis.edu

Description:
	Split the knowledge base into multiple folds for evaluation.

To-do:
	1. Make it generalizable by removing gene specific items.
"""

import os
import numpy as np
import pandas as pd
import logging as log
from sklearn.model_selection import KFold

class SplitFolds():
	"""
	Class for splitting the knowledge base into folds.
	"""

	# class variables (need to be removed later)
	COLUMNS = ['Subject', 'Predicate', 'Object', 'Label']
	CRTA_STR = 'confers resistance to antibiotic'

	def __init__(self, pd_data, num_folds):
		"""
		Class constructor for SplitFolds.

		Inputs:
			pd_data: integrated data where
				pd_data.columns.values = ['Subject' 'Predicate' 'Object' 'Label']
			num_folds: number of folds to split the data into
		"""
		self.pd_data = pd_data
		self.num_folds = num_folds

	def split_into_folds(self, genes):
		"""
		Perform data split.

		Inputs:
			genes: numpy array containing unique entities
				(to be used for closed-world assumption)

		Returns:
			data_split_fold_dic: dictionary where the key is
				train / dev / test for each fold and the value
				is a dataframe containing the splitted
				knowledge graph
		"""
		data_split_fold_dic = {}

		# generate a pool to be used later in random sampling assuming a closed world
		cra_genes = self.pd_data[self.pd_data['Predicate'].isin([self.CRTA_STR])]['Subject'].unique()
		genes_pool_closed_world = np.array(list(set(genes.tolist()) - set(cra_genes.tolist())))

		# extract pos / neg data based on the label
		pos_data = self.pd_data[self.pd_data['Label'] == '1']
		neg_data = self.pd_data[self.pd_data['Label'] == '-1']

		# positive data except CRA edge
		pos_data_cra_only = pos_data[pos_data['Predicate'].isin([self.CRTA_STR])]
		pos_data_except_cra = pos_data[~pos_data['Predicate'].isin([self.CRTA_STR])]

		# negative data with only CRA edge
		neg_data_cra_only = neg_data[neg_data['Predicate'].isin([self.CRTA_STR])]

		# allocate 80% of positive CRA edges to allocate to train & dev
		num_pos_cra_train_dev = int(pos_data_cra_only.shape[0] * 0.08)

		# distribute CRA edges among train / dev / test for specified folds
		k = 0
		for train_dev_index, test_index in KFold(n_splits=self.num_folds).split(pos_data_cra_only):
			np.random.shuffle(train_dev_index)

			train_index = train_dev_index[num_pos_cra_train_dev:]
			dev_index = train_dev_index[0:num_pos_cra_train_dev]

			data_split_fold_dic['fold_{}_train'.format(k)] = pos_data_cra_only.iloc[train_index, :]
			data_split_fold_dic['fold_{}_dev_without_neg'.format(k)] = pos_data_cra_only.iloc[dev_index, :]
			data_split_fold_dic['fold_{}_test_without_neg'.format(k)] = pos_data_cra_only.iloc[test_index, :]

			k += 1

		# fill up train with other positive predicates for all folds
		for k in range(self.num_folds):
			data_split_fold_dic['fold_{}_train'.format(k)] = \
				data_split_fold_dic['fold_{}_train'.format(k)].append(pos_data_except_cra)

		# need to do random sampling to generate 49 negatives for each positive
		for k in range(self.num_folds):
			log.info('Processing fold: {}'.format(k))

			data_split_fold_dic = self._random_sample_negs(
				k, data_split_fold_dic, neg_data_cra_only, genes_pool_closed_world, 'dev')

			data_split_fold_dic = self._random_sample_negs(
				k, data_split_fold_dic, neg_data_cra_only, genes_pool_closed_world, 'test')

		for key, value in data_split_fold_dic.items() :
			log.debug('Shape of {}: {}'.format(key, value.shape[0]))

		return data_split_fold_dic

	def save_folds(self, data_split_fold_dic, save_parent_dir):
		"""
		Save the processed folds into the specified directory.
		save_parent_dir / folds / fold_0
								/ fold_1
						.
						.
						.
								/ fold_k

		Inpus:
			data_split_fold_dic: dictionary processed in 'split_into_folds()'
			save_parent_dir: parent directory to save the folds into
		"""
		sub_parent_dir = os.path.join(save_parent_dir, 'folds')

		for k in range(self.num_folds):
			each_fold_parent_dir = os.path.join(sub_parent_dir, 'fold_{}'.format(k))

			if not os.path.exists(each_fold_parent_dir):
				os.makedirs(each_fold_parent_dir)

			data_split_fold_dic['fold_{}_train'.format(k)].to_csv(os.path.join(each_fold_parent_dir, 'train.txt'), sep='\t', index=False, header=None)
			data_split_fold_dic['fold_{}_dev'.format(k)].to_csv(os.path.join(each_fold_parent_dir, 'dev.txt'), sep='\t', index=False, header=None)
			data_split_fold_dic['fold_{}_test'.format(k)].to_csv(os.path.join(each_fold_parent_dir, 'test.txt'), sep='\t', index=False, header=None)

	def _random_sample_negs(self, cur_fold, data_split_fold_dic, neg_data_cra_only, closed_world_sub_pool, dtype, num_negs=49):
		"""
		(Private) Randomly 'num_negs' known negatives for each positive.
		If there are not enough known negatives, randomly sample negatives
		using closed-world assumption.

		Inputs:
			cur_fold: fold # currently being processed
			data_split_fold_dic: dictionary processed in 'split_into_folds()'
			neg_data_cra_only: negatives where the relation is only CRTA
			closed_world_sub_pool: pool of subjects assuming a closed world
			dtype: data type (i.e. 'dev' or 'test')
			num_negs: number of negatives to sample per positive

		Returns:
			data_split_fold_dic: updated dictionary containing negatives
		"""

		# init
		new_with_neg_cra = pd.DataFrame(columns=self.COLUMNS)

		# find how many total positives there are
		pos_size = data_split_fold_dic['fold_{}_{}_without_neg'.format(cur_fold, dtype)].shape[0]
		log.debug('Size of fold {} {} only positives: {}'.format(cur_fold, dtype, pos_size))

		# for each positive, randomly sample 'num_negs' negatives and append to 'new_with_neg_cra'
		for i in range(pos_size):
			# actual positive SPO that we're working on
			pos_spo = data_split_fold_dic['fold_{}_{}_without_neg'.format(cur_fold, dtype)].iloc[i, :]
			sub = pos_spo['Subject']
			obj = pos_spo['Object']

			# find negative samples which has same object as the positive SPO
			# but different subject than that in positive SPO
			neg_samples = neg_data_cra_only[neg_data_cra_only['Object'].isin([obj])]
			neg_samples = neg_samples[~neg_samples['Subject'].isin([sub])]

			# shuffle the negative samples in case there are more than 1 negative samples,
			if neg_samples.shape[0] > 0:
				neg_samples = neg_samples.sample(frac=1).reset_index(drop=True)

			# append the original positive SPO
			new_with_neg_cra = new_with_neg_cra.append(pos_spo)

			# append the randomly sampled negative SPOs after the positive SPO
			if neg_samples.shape[0] > num_negs: # all negatives can be sample from known negatives
				new_with_neg_cra = new_with_neg_cra.append(neg_samples.iloc[0:num_negs, :])
			else: # not all negatives can be sample from known negatives
				# find how many negatives has to be added assuming closed world
				num_additional_subjs = num_negs - neg_samples.shape[0]
				selected_subjs = np.random.choice(closed_world_sub_pool, num_additional_subjs, replace=False)
				log.debug('Number of additional subjects: {}'.format(num_additional_subjs))

				# if we still have some known negatives, append it now
				if neg_samples.shape[0] > 0:
					new_with_neg_cra = new_with_neg_cra.append(neg_samples)

				# fill rest of the space with randomly samples negatives from closed world assumption
				random_generated_neg_cra = pd.DataFrame(0, index=np.arange(num_additional_subjs), columns=self.COLUMNS)
				random_generated_neg_cra['Subject'] = selected_subjs
				random_generated_neg_cra['Predicate'] = self.CRTA_STR
				random_generated_neg_cra['Object'] = obj
				random_generated_neg_cra['Label'] = '-1'

				new_with_neg_cra = new_with_neg_cra.append(random_generated_neg_cra)

		log.debug('Newly sampled {} with negative samples: {}'.format(dtype, new_with_neg_cra.shape[0]))

		data_split_fold_dic['fold_{}_{}'.format(cur_fold, dtype)] = new_with_neg_cra.reset_index(drop=True)

		return data_split_fold_dic
