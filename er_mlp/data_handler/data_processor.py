"""
Filename: data_processor.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu

Description:
	Class containing multiple functions that will be used to process the data.

To-do:
"""

import sys
import re
import random
import numpy as np
import pandas as pd
import scipy.io as spio

class DataProcessor:
	"""
	Collection of functions to process the data.
	"""

	def load(self, filename):
		"""
		Load the data and return the pandas dataframe.

		Inputs:
			filename: filename containing the full path

		Returns:
			df: dataframe containing the data
		"""
		df = pd.read_csv(filename, sep='\t', encoding ='latin-1', header=None)

		return df

	def create_dic_index(self,dic):
		dictionary = {}
		index = 0
		for k,v in dic.items():
			dictionary[k] = index
			index += 1
		return dictionary

	def create_indexed_triplets_training(self, df_data, entity_dic, pred_dic):
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

	def create_indexed_triplets_test(self, df_data, entity_dic, pred_dic):
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
			[sub_index, pred_index, obj_index, true/false]
		"""
		indexed_data = [[entity_dic[df_data[i][0]], pred_dic[df_data[i][1]], entity_dic[df_data[i][2]], df_data[i][3]] for i in range(len(df_data))]

		return np.array(indexed_data)

	def create_entity_dic(self, training_data):
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

	def create_relation_dic(self, training_data):
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

	def machine_translate_using_word(self, fname, initEmbedFile=None, separator='_'):
		"""
		Given a file containing either entities or relations, translate them into
		machine friendly setting using words. For example, assume we are given two entities
		'A_C' and 'B_C'. First, separate them into words and generate word pool 'A', 'B', and 'C'.
		Then represent each entity as combination of these words.

		Inputs:
			fname: filename containing all the entities / relations
			initEmbedFile: (optional) mat file to use instead
			separator: (optional) separator which separates words within an entity / relation

		Returns:
			indexed_items: list of lists [[], [], []] where each list
				contains word index ids for a single entity / relation
			num_words: total number of words inside the all the entities / relations
			item_dic: dictionary whose key is entity / relation and
				value is the index assigned to that entity / relation
		"""
		item_dic = {}
		item_index_id = 0

		with open(fname, encoding='utf8') as f:
			items = [l.split() for l in f.read().strip().split('\n')]

		# create item dictionary where key is item
		# and value is the index assigned to that item
		for e in items:
			if e[0] not in item_dic:
				item_dic[e[0]] = item_index_id
				item_index_id += 1

		if initEmbedFile:
			print('using init embed file')
			mat = spio.loadmat(initEmbedFile, squeeze_me=True)
			indexed_items = [[mat['tree'][i][()][0] - 1] if isinstance(mat['tree'][i][()][0],int) else [x - 1 for x in mat['tree'][i][()][0]] for i in range(len(mat['tree'])) ]
			num_words = len(mat['words'])
		else:
			word_index_id = 0
			word_ids = {}
			items_to_words = {}

			# for each item
			for e in items:
				words = []

				# e[0] = molecular_function
				for s in re.split(separator, e[0]):
					words.append(s)

					if s not in word_ids:
						word_ids[s] = word_index_id
						word_index_id += 1

				# words = ['molecular', 'function']
				items_to_words[e[0]] = words

			# create list of of length len(item_dic) where all entries are None
			indexed_items = [None] * len(item_dic)

			for key, val in item_dic.items():
				indexed_items[val] = []

				# items_to_words[key] = ['molecular', 'function']
				for s in items_to_words[key]:
					indexed_items[val].append(word_ids[s])

			num_words = len(word_ids)

		return indexed_items, num_words, item_dic

	def machine_translate(self, fname):
		"""
		Given a file containing entities, assign each entity
		with unique entity index id which machine can use.

		Inputs:
			fname: filename containing all the entities

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
