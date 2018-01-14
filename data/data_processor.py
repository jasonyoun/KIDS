import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio

class DataProcessor:
    
    def load(self,filename):
        df = pd.read_csv(filename,sep='\t',encoding ='latin-1',header=0)
        return df

    def create_dic_index(self,dic):
        dictionary = {}
        index = 0
        for k,v in dic.items():
            dictionary[k] = index
            index += 1
        return dictionary

    def create_indexed_triplets_training(self,training_data,entity_dic,pred_dic):
        indexed_data = [[entity_dic[training_data[i][0]], pred_dic[training_data[i][1]], entity_dic[training_data[i][2]]] for i in range(len(training_data))]
        return np.array(indexed_data)

    def create_indexed_triplets_test(self,training_data,entity_dic,pred_dic):
        indexed_data = [[entity_dic[training_data[i][0]], pred_dic[training_data[i][1]], entity_dic[training_data[i][2]], training_data[i][3]] for i in range(len(training_data))]
        return np.array(indexed_data)

    def machine_translate_using_word(self,fname,embedding_size, initEmbedFile=None):
        f = open(fname, encoding='utf8')
        entities = [l.split() for l in f.read().strip().split('\n')]
        f.close()
        entity_dic = {}
        num_words = None       
        entity_index_id = 0
        for e in entities:
            #e[0] = e[0].lstrip("_")
            if e[0] not in entity_dic:
                entity_dic[e[0]] = entity_index_id
                entity_index_id+=1
        indexed_entities =None
        if (initEmbedFile):
            mat = spio.loadmat(initEmbedFile, squeeze_me=True)
            indexed_entities = [ [mat['tree'][i][()][0] - 1] if isinstance(mat['tree'][i][()][0],int) else [x - 1 for x in mat['tree'][i][()][0]] for i in range(len(mat['tree'])) ]
            num_words = len(mat['words'])
        else:
            word_index_id = 0
            word_ids = {}
            entities_to_words = {}
            for e in entities:
                #e[0] = e[0].lstrip("_")
                words = []
                for s in e[0].split('_'):
                    words.append(s)
                    if s not in word_ids:
                        word_ids[s] = word_index_id
                        word_index_id +=1
                entities_to_words[e[0]] = words

            indexed_entities =[None]*len(entity_dic)
            for k,v in entity_dic.items():
                indexed_entities[v] = []
                for s in entities_to_words[k]:
                    indexed_entities[v].append(word_ids[s])
            num_words = len(word_ids)
        return indexed_entities, num_words, entity_dic

    def machine_translate(self,fname,embedding_size):
        f = open(fname, encoding='utf8')
        entities = [l.split() for l in f.read().strip().split('\n')]
        f.close()
        entity_index_id = 0
        entity_dic = {}
        for e in entities:
            if e[0] not in entity_dic:
                entity_dic[e[0]] = entity_index_id
                entity_index_id+=1

        return entity_dic










