"""
Filename: extract_info.py

Authors:
	Jason Youn -jyoun@ucdavis.edu

Description:
	Given a integrated knowledge base, extract information
	like entities / relations and create set of functions
	to access these information.

To-do:
"""

import numpy as np
import pandas as pd
import logging as log

class ExtractInfo():
	"""
	Class for extracting information from the integrated dataset.
	"""

	# class variables (need to be removed later)
	SUBJECT_STR = 'Subject'
	PRED_STR = 'Predicate'
	OBJECT_STR = 'Object'

	DOMAIN_STR = 'Domain'
	RELATION_STR = 'Relation'
	RANGE_STR = 'Range'

	def __init__(self, pd_data, drr_path):
		"""
		Class constructor for ExtractInfo.

		Inputs:
			pd_data: integrated data where
				pd_data.columns.values = ['Subject' 'Predicate' 'Object' 'Label']
			drr_path: path to the text file which contains the
				domain / relation / range relationship
		"""
		self.pd_data = pd_data
		self.pd_drr, self.relations, self.entity_types = self._read_drr(drr_path)
		self.entity_dic = self._fill_entity_dic()

	def get_entity_by_type(self, entity_type):
		"""
		Get set of entities by specifying the type of entity.

		Inputs:
			entity_type: type of entity (e.g. 'gene', 'antibiotic')

		Returns:
			numpy array containing the entities matching the type
		"""
		assert entity_type in list(self.entity_dic.keys())

		return self.entity_dic[entity_type]

	def save_all_entities(self, file_path):
		"""
		Save all the entities to the specified file location.

		Inputs:
			file_path: file path to save all the entities
		"""
		log.info('Saving entities to \'{}\'...'.format(file_path))

		all_entities = np.array([])

		for entity_type in list(self.entity_dic.keys()):
			all_entities = np.append(all_entities, self.entity_dic[entity_type])

		all_entities = np.unique(all_entities)

		np.savetxt(file_path, all_entities, fmt='%s')

	def _read_drr(self, drr_path, get_overlap_only=True):
		"""
		(Private) Read the domain / relation / range text file.

		Inputs:
			drr_path: file path of the text file containing the
				domain / relation / range information
			get_overlap_only: True if only getting the DRR
				info that is available in the dataset
				(used to skip negative relations)

		Returns:
			pd_drr: dataframe containing the DRR info
			all_relations_list: list containing all the relations
			entity_types: list containing all the unique entity types
		"""
		pd_drr = pd.read_csv(drr_path, sep = '\t')

		# get all the relations in the dataset working on
		relation_group = self.pd_data.groupby(self.PRED_STR)
		all_relations_list = list(relation_group.groups.keys())

		# find unique entity types
		entity_types = []
		for dr_tuple, _ in pd_drr.groupby([self.DOMAIN_STR, self.RANGE_STR]):
			entity_types.extend(list(dr_tuple))
		entity_types = list(set(entity_types))

		# get only the (domain / relation / range) data
		# that is available in the dataset
		if get_overlap_only:
			matching_relations_idx = pd_drr[self.RELATION_STR].isin(all_relations_list)
			pd_drr = pd_drr[matching_relations_idx].reset_index(drop=True)

		log.debug('(Domain / Relation / Range) to process:\n{}'.format(pd_drr))

		return pd_drr, all_relations_list, entity_types

	def _fill_entity_dic(self):
		"""
		(Private) Complete the dictionary where the key is entity type
		and value is the numpy array containing all the entities belonging
		to that type.

		Returns:
			entity_dic: completed entity dictionary
		"""
		entity_dic = dict.fromkeys(self.entity_types, np.array([]))

		# loop through each relation and fill in the
		# entity dictionary based on their domain / range type
		for relation in self.relations:
			log.debug('Processing relation: {}'.format(relation))
			single_grr = self.pd_drr.loc[self.pd_drr[self.RELATION_STR] == relation]

			domain_type = single_grr[self.DOMAIN_STR].item()
			range_type = single_grr[self.RANGE_STR].item()

			domains, ranges = self._get_domain_range(relation)

			entity_dic[domain_type] = np.append(entity_dic[domain_type], domains)
			entity_dic[range_type] = np.append(entity_dic[range_type], ranges)

		for key, value in entity_dic.items():
			entity_dic[key] = np.unique(value)
			log.debug('Count of entity type \'{}\': {}'.format(key, entity_dic[key].shape[0]))

		return entity_dic

	def _get_domain_range(self, relation):
		"""
		Given a relation, find all the unique domains and ranges.

		Inputs:
			relation: relation type

		Returns:
			all_unique_domains: all unique domains
			all_unique_ranges: all unique ranges
		"""
		# get index which has specified relation type
		matching_relation_idx = self.pd_data[self.PRED_STR].isin([relation])

		# find unique domain & range for given relation type
		all_unique_domains = self.pd_data[matching_relation_idx][self.SUBJECT_STR].unique()
		all_unique_ranges = self.pd_data[matching_relation_idx][self.OBJECT_STR].unique()

		return all_unique_domains, all_unique_ranges
