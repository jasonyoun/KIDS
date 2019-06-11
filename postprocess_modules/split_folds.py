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
import sys
import logging as log
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class SplitFolds():
    """
    Class for splitting the knowledge base into folds.
    """

    # class variables (need to be removed later)
    COLUMNS = ['Subject', 'Predicate', 'Object', 'Label']
    CRTA_STR = 'confers resistance to antibiotic'

    def __init__(self, pd_data, num_folds, all_genes):
        """
        Class constructor for SplitFolds.

        Inputs:
            pd_data: integrated data where
                pd_data.columns.values = ['Subject' 'Predicate' 'Object' 'Label']
            num_folds: number of folds to split the data into
            all_genes: numpy array containing unique entities
                (to be used for closed-world assumption)
        """
        self.pd_data = pd_data.sample(frac=1).reset_index(drop=True)
        self.num_folds = num_folds
        self.all_genes = all_genes

        self._extract_cra()

        # appened to the cra only negatives with synthetically generated negatives
        self.updated_neg_data_cra_only, num_pos_to_remove_dic = self._generate_synthetic_negs()

        # we have some positives to remove to balance pos and neg
        if num_pos_to_remove_dic:
            self._update_data(num_pos_to_remove_dic)
            self._extract_cra()

    def _update_data(self, num_pos_to_remove_dic):

        for obj, count in num_pos_to_remove_dic.items():
            indices = self.pos_data_cra_only.index[self.pos_data_cra_only['Object'].isin([obj])].tolist()
            chosen_indices = indices[0:count]
            self.pd_data = self.pd_data.drop(self.pd_data.index[chosen_indices])

        # self.pd_data = self.pd_data.reset_index(drop=True)
        self.pd_data = self.pd_data.sample(frac=1).reset_index(drop=True)

    def _extract_cra(self):
        self.all_cra = self.pd_data[self.pd_data['Predicate'].isin([self.CRTA_STR])]

        # extract pos / neg data based on the label
        pos_data = self.pd_data[self.pd_data['Label'].astype(str) == '1']
        neg_data = self.pd_data[self.pd_data['Label'].astype(str) == '-1']

        # positive data with only or without any CRA edge
        self.pos_data_cra_only = pos_data[pos_data['Predicate'].isin([self.CRTA_STR])]
        self.pos_data_except_cra = pos_data[~pos_data['Predicate'].isin([self.CRTA_STR])]

        # negative data with only CRA edge
        self.neg_data_cra_only = neg_data[neg_data['Predicate'].isin([self.CRTA_STR])]

    def split_into_folds(self):
        """
        Perform data split.

        Returns:
            data_split_fold_dic: dictionary where the key is
                train / dev / test for each fold and the value
                is a dataframe containing the splitted
                knowledge graph
        """
        data_split_fold_dic = {}

        # distribute CRA edges among train / dev / test for specified folds
        k = 0
        for train_dev_index, test_index in KFold(n_splits=self.num_folds).split(self.pos_data_cra_only):
            np.random.shuffle(train_dev_index)

            # allocate 90% of train_dev_index into train
            num_train = int(0.9 * train_dev_index.shape[0])
            train_index = train_dev_index[0:num_train]
            dev_index = train_dev_index[num_train:]

            data_split_fold_dic['fold_{}_train'.format(k)] = self.pos_data_cra_only.iloc[train_index, :]
            data_split_fold_dic['fold_{}_train_local_without_neg'.format(k)] = self.pos_data_cra_only.iloc[train_index, :]
            data_split_fold_dic['fold_{}_dev_without_neg'.format(k)] = self.pos_data_cra_only.iloc[dev_index, :]
            data_split_fold_dic['fold_{}_test_without_neg'.format(k)] = self.pos_data_cra_only.iloc[test_index, :]

            k += 1

        # fill up train with other positive predicates for all folds
        for k in range(self.num_folds):
            data_split_fold_dic['fold_{}_train'.format(k)] = \
                data_split_fold_dic['fold_{}_train'.format(k)].append(self.pos_data_except_cra)

        # need to do random sampling to generate 49 negatives for each positive
        for k in range(self.num_folds):
            log.info('Processing fold: %d', k)

            data_split_fold_dic = self._random_sample_negs(k, data_split_fold_dic, 'train_local')
            data_split_fold_dic = self._random_sample_negs(k, data_split_fold_dic, 'dev')
            data_split_fold_dic = self._random_sample_negs(k, data_split_fold_dic, 'test')

        for key, value in data_split_fold_dic.items():
            log.debug('Shape of %s: %d', key, value.shape[0])

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
            data_split_fold_dic['fold_{}_train_local'.format(k)].to_csv(os.path.join(each_fold_parent_dir, 'train_local.txt'), sep='\t', index=False, header=None)
            data_split_fold_dic['fold_{}_dev'.format(k)].to_csv(os.path.join(each_fold_parent_dir, 'dev.txt'), sep='\t', index=False, header=None)
            data_split_fold_dic['fold_{}_test'.format(k)].to_csv(os.path.join(each_fold_parent_dir, 'test.txt'), sep='\t', index=False, header=None)

    def _generate_synthetic_negs(self, num_negs=49):
        num_pos_to_remove_dic = {}
        random_sampled_negs = pd.DataFrame(columns=self.COLUMNS)

        objs_values = self.pos_data_cra_only['Object'].value_counts().keys().tolist()
        objs_counts = self.pos_data_cra_only['Object'].value_counts().tolist()
        objs_value_counts_dic = dict(zip(objs_values, objs_counts))

        for obj, count in objs_value_counts_dic.items():
            num_negs_needed = num_negs * count
            num_known_neg_samples = self.neg_data_cra_only[self.neg_data_cra_only['Object'].isin([obj])].shape[0]

            # if we don't have enough known negatives to sample for given object
            if num_known_neg_samples < num_negs_needed:
                cra_same_obj = self.all_cra[self.all_cra['Object'].isin([obj])]
                known_genes = cra_same_obj['Subject'].unique()
                random_picked_subjs = np.array(list(set(self.all_genes.tolist()) - set(known_genes.tolist())))

                # find how many random subjects we need
                num_random_subjs = num_negs_needed - num_known_neg_samples

                log.debug('Number of randomly sampled subjects for %s: %d', obj, num_random_subjs)

                if random_picked_subjs.shape[0] < num_random_subjs:
                    log.warning('Not enough negatives to randomly pick for %s', obj)

                    num_pos_to_remove_dic[obj] = int(np.ceil((num_random_subjs - random_picked_subjs.shape[0]) / num_negs))
                    num_random_subjs -= (num_negs * num_pos_to_remove_dic[obj])

                # fill randomly samples negatives using closed world assumption
                random_generated_neg_cra = pd.DataFrame(0, index=np.arange(num_random_subjs), columns=self.COLUMNS)
                random_generated_neg_cra['Subject'] = np.random.choice(random_picked_subjs, num_random_subjs, replace=False)
                random_generated_neg_cra['Predicate'] = self.CRTA_STR
                random_generated_neg_cra['Object'] = obj
                random_generated_neg_cra['Label'] = '-1'

                random_sampled_negs = random_sampled_negs.append(random_generated_neg_cra)

        updated_neg_data_cra_only = pd.concat([self.neg_data_cra_only, random_sampled_negs])
        updated_neg_data_cra_only = updated_neg_data_cra_only.sample(frac=1).reset_index(drop=True)

        return updated_neg_data_cra_only, num_pos_to_remove_dic

    def _random_sample_negs(self, cur_fold, data_split_fold_dic, dtype, num_negs=49):
        """
        (Private) Randomly 'num_negs' known negatives for each positive.
        If there are not enough known negatives, randomly sample negatives
        using closed-world assumption.

        Inputs:
            cur_fold: fold # currently being processed
            data_split_fold_dic: dictionary processed in 'split_into_folds()'
            dtype: data type (i.e. 'dev' | 'test' | 'train_local')
            num_negs: number of negatives to sample per positive

        Returns:
            data_split_fold_dic: updated dictionary containing negatives
        """

        # init
        known_negatives_dic = {}
        new_with_neg_cra = pd.DataFrame(columns=self.COLUMNS)

        # find how many total positives there are
        pos_size = data_split_fold_dic['fold_{}_{}_without_neg'.format(cur_fold, dtype)].shape[0]
        log.debug('Size of fold %d %s only positives: %d', cur_fold, dtype, pos_size)

        # for each positive, randomly sample 'num_negs' negatives and append to 'new_with_neg_cra'
        for i in range(pos_size):
            # actual positive SPO that we're working on
            pos_spo = data_split_fold_dic['fold_{}_{}_without_neg'.format(cur_fold, dtype)].iloc[i, :]
            obj = pos_spo['Object']

            # append the original positive SPO
            new_with_neg_cra = new_with_neg_cra.append(pos_spo)

            # before, append negatives find known negatives
            # and save to the dictionary if it already does not exist
            if obj not in known_negatives_dic:
                # find negative samples which has same object as the positive SPO
                # but different subject than that in positive SPO
                known_negatives_dic[obj] = self.updated_neg_data_cra_only[self.updated_neg_data_cra_only['Object'].isin([obj])]

            if known_negatives_dic[obj].shape[0] < num_negs:
                sys.exit('We are supposed to have enough negatives now!')

            new_with_neg_cra = new_with_neg_cra.append(known_negatives_dic[obj].iloc[0:num_negs, :])

            # remove used known negatives
            known_negatives_dic[obj] = known_negatives_dic[obj].iloc[num_negs:, :]

        log.debug('Newly sampled %s with negative samples: %d', dtype, new_with_neg_cra.shape[0])

        data_split_fold_dic['fold_{}_{}'.format(cur_fold, dtype)] = new_with_neg_cra.reset_index(drop=True)

        return data_split_fold_dic
