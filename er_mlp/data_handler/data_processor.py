"""
Filename: data_processor.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Class containing multiple functions that will be used to process the data.

To-do:
"""
import re
import logging as log
import numpy as np
import pandas as pd
import scipy.io as spio

class DataProcessor:
    """
    Collection of functions to process the data.
    """
    def __init__(self):
        """
        Constructor for DataProcessor class.
        """

    @staticmethod
    def load(filename):
        """
        Load the data and return the pandas dataframe.

        Inputs:
            filename: filename containing the full path

        Returns:
            pd_data: dataframe containing the data
        """
        log.info('Loading data from \'%s\'...', filename)
        pd_data = pd.read_csv(
            filename, sep='\t', encoding='latin-1',
            names=['Subjct', 'Predicate', 'Object', 'Label'])

        return pd_data

    @staticmethod
    def create_dic_index(dic):
        dictionary = {}
        index = 0
        for k, _ in dic.items():
            dictionary[k] = index
            index += 1
        return dictionary

    @staticmethod
    def create_indexed_triplets_training(df_data, entity_dic, pred_dic):
        """
        Same as self.create_indexed_triplets_test() except for the lack
        of true/false field in the returning numpy array.

        Inputs:
            df_data: dataframe containing the data
            entity_dic: dictionary of entities where the key is entity,
                and the value is a index id assigned to that specific entity.
            pred_dic: dictionary of entities where the key is pred,
                and the value is a index id assigned to that specific pred.

        Returns:
            list of lists where each list has length equal to 3.
            [sub_index, pred_index, obj_index]
        """
        indexed_data = [[entity_dic[df_data[i][0]], pred_dic[df_data[i][1]], entity_dic[df_data[i][2]]] for i in range(len(df_data))]

        return np.array(indexed_data)

    @staticmethod
    def create_indexed_triplets_test(df_data, entity_dic, pred_dic):
        """
        Given a train / dev / test dataset, create a numpy array
        which consists of indeces of all the items in the triple.

        Inputs:
            df_data: dataframe containing the data
            entity_dic: dictionary of entities where the key is entity,
                and the value is a index id assigned to that specific entity.
            pred_dic: dictionary of entities where the key is pred,
                and the value is a index id assigned to that specific pred.

        Returns:
            list of lists where each list has length equal to 4.
            [sub_index, pred_index, obj_index, 1/-1]
        """
        indexed_data = [[entity_dic[df_data[i][0]], pred_dic[df_data[i][1]], entity_dic[df_data[i][2]], df_data[i][3]] for i in range(len(df_data))]

        return np.array(indexed_data)

    @staticmethod
    def create_entity_dic(training_data):
        entity_set = set()

        for i in range(len(training_data)):
            entity_set.add(training_data[i][0])
            entity_set.add(training_data[i][2])

        dic = {}
        index_id = 0

        for entity in entity_set:
            if entity not in dic:
                dic[entity] = index_id
                index_id += 1

        return dic

    @staticmethod
    def create_relation_dic(training_data):
        dic = {}
        relation_set = set()

        for i in range(len(training_data)):
            relation_set.add(training_data[i][1])

        index_id = 0

        for relation in relation_set:
            if relation not in dic:
                dic[relation] = index_id
                index_id += 1

        return dic

    @staticmethod
    def machine_translate_using_word(fname, init_embed_file=None, separator='_'):
        """
        Given a file containing either entities or relations, translate them into
        machine friendly setting using words. For example, assume we are given two entities
        'A_C' and 'B_C'. First, separate them into words and generate word pool 'A', 'B', and 'C'.
        Then represent each entity as combination of these words.

        Inputs:
            fname: filename containing all the entities / relations
            init_embed_file: (optional) mat file to use instead
            separator: (optional) separator which separates words within an entity / relation

        Returns:
            indexed_items: list of lists [[], [], []] where each list
                contains word index ids for a single entity / relation
            num_words: total number of words inside the all the entities / relations
            item_dic: dictionary whose key is entity / relation and
                value is the index assigned to that entity / relation
        """
        # some inits
        item_dic = {}
        item_index_id = 0

        log.info('Performing machine translation for items in \'%s\' using word embedding...', fname)

        # open file
        with open(fname, encoding='utf8') as _file:
            items = [l.split() for l in _file.read().strip().split('\n')]

        # create item dictionary where key is item
        # and value is the index assigned to that item
        for item in items:
            if item[0] not in item_dic:
                item_dic[item[0]] = item_index_id
                item_index_id += 1

        if init_embed_file:
            log.info('Using init embed file \'%s\' for word embedding...', init_embed_file)
            mat = spio.loadmat(init_embed_file, squeeze_me=True)
            indexed_items = [[mat['tree'][i][()][0] - 1] if isinstance(mat['tree'][i][()][0], int) else [x - 1 for x in mat['tree'][i][()][0]] for i in range(len(mat['tree']))]
            num_words = len(mat['words'])
        else:
            word_index_id = 0
            word_ids = {}
            items_to_words = {}

            # for each item
            for item in items:
                words = []

                # item[0] = molecular_function
                for s in re.split(separator, item[0]):
                    words.append(s)

                    if s not in word_ids:
                        word_ids[s] = word_index_id
                        word_index_id += 1

                # words = ['molecular', 'function']
                items_to_words[item[0]] = words

            # create list of of length len(item_dic) where all entries are None
            indexed_items = [None] * len(item_dic)

            for key, val in item_dic.items():
                indexed_items[val] = []

                # items_to_words[key] = ['molecular', 'function']
                for s in items_to_words[key]:
                    indexed_items[val].append(word_ids[s])

            num_words = len(word_ids)

        return indexed_items, num_words, item_dic

    @staticmethod
    def machine_translate(fname):
        """
        Given a file containing entities, assign each entity
        with unique entity index id which machine can use.

        Inputs:
            fname: filename containing all the entities

        Returns:
            entity_dic: dictionary of entities where the key is entity,
            and the value is a index id assigned to that specific entity.
        """
        # some inits
        entity_dic = {}
        entity_index_id = 0

        log.info('Performing machine translation for items in \'%s\' without using word embedding...', fname)

        # open file
        with open(fname, encoding='utf-8') as _file:
            entities = [l.split() for l in _file.read().strip().split('\n')]

        for entity in entities:
            if entity[0] not in entity_dic:
                entity_dic[entity[0]] = entity_index_id
                entity_index_id += 1

        return entity_dic
