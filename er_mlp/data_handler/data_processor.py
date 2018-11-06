import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio

class DataProcessor:

	def load(self,filename):
		df = pd.read_csv(filename,sep='\t',encoding ='latin-1',header=None)
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

	def create_indexed_triplets_test(self, raining_data, entity_dic, pred_dic):
		indexed_data = [[entity_dic[training_data[i][0]], pred_dic[training_data[i][1]], entity_dic[training_data[i][2]], training_data[i][3]] for i in range(len(training_data))]
		return np.array(indexed_data)

	def create_entity_dic(self,training_data):
		entity_set = set()
		for i in range(len(training_data)):
			entity_set.add(training_data[i][0])
			entity_set.add(training_data[i][2])
		dic = {}
		index_id = 0
		for e in entity_set:
			if e not in dic:
				dic[e] = index_id
				index_id+=1
		return dic

	def create_relation_dic(self,training_data):
		dic = {}
		relation_set = set()
		for i in range(len(training_data)):
			relation_set.add(training_data[i][1])
		index_id = 0
		for r in relation_set:
			if r not in dic:
				dic[r] = index_id
				index_id+=1
		return dic

	def machine_translate_using_word(self,fname,embedding_size, initEmbedFile=None, separator='_'):
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
			print('initembed')
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
				for s in re.split(separator,e[0]):
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

	def machine_translate(self, fname, embedding_size):
		"""
		Given a file containing entities, assign each entity
		with unique entity index id which machine can use.

		Inputs:
			fname: filename containing all the entities
			embedding_size: 	

		Returns:
			entity_dic: dictionary of entities where the key is entity,
			and the value is a index id assigned to that specific entity.
		"""
		entity_dic = {}
		entity_index_id = 0

		with open(fname, encoding='utf-8') as f:
			entities = [l.split() for l in f.read().strip().split('\n')]

		for e in entities:
			if e[0] not in entity_dic:
				entity_dic[e[0]] = entity_index_id
				entity_index_id += 1

		return entity_dic
