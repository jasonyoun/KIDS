"""
Filename: inconsistency_manager.py

Authors:
	Minseung Kim - msgkim@ucdavis.edu

Description:
	Detect the inconsistencies.

To-do:
	1. Move _SPO_LIST to global file.
"""

import operator
import math
import numpy as np
import pandas as pd
import logging as log
from collections import Counter
from .utilities import get_pd_of_statement
import xml.etree.ElementTree as ET

class InconsistencyManager:
	"""
	Class for detecting the inconsistencies.
	"""

	_SPO_LIST = ['Subject', 'Predicate', 'Object']

	def __init__(self, inconsistency_rule_file):
		"""
		Class constructor foro InconsistencyManager.

		Inputs:
			inconsistency_rule_file: XML file name containing the inconsistency rules
		"""
		self.inconsistency_rule_file = inconsistency_rule_file

	def detect_inconsistencies(self, pd_data):
		"""
		Detect the inconsistencies among the data using the provided rule file.

		Inputs:
			pd_data: data to detect inconsistency from

					       Subject     Predicate Object Source
					0          lrp  no represses   fadD  hiTRN
					1          fur  no represses   yfeD  hiTRN
					2          fnr  no represses   ybhN  hiTRN
					3          crp  no represses   uxuR  hiTRN

		Returns:
			inconsistencies: Dictionary containing inconsistency_id as key
				and list of inconsistent triples + source as value

				{
				0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
					(('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
				1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
					('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
										...
				}
		"""
		inconsistency_rules = ET.parse(self.inconsistency_rule_file).getroot()
		inconsistencies = {}
		inconsistency_id = 0

		log.info('Detecting inconsistencies using inconsisty rule {}'.format(self.inconsistency_rule_file))

		# iterate over each inconsistency rule
		for inconsistency_rule in inconsistency_rules:
			inconsistency_rule_name = inconsistency_rule.get('name')
			condition_statement = inconsistency_rule.find('condition')

			log.debug('Processing inconsisency rule \'{}\''.format(inconsistency_rule_name))

			# check if condition is met
			pd_condition_specific_data = pd_data.copy()

			if condition_statement is not None:
				pd_condition_statement = get_pd_of_statement(condition_statement)
				indices_meeting_condition = (pd_data[pd_condition_statement.index] == pd_condition_statement).all(1).values
				pd_condition_specific_data = pd_data[indices_meeting_condition].copy()

				# skip to the next rule if condition statement exists and there are no data meeting the condition
				if pd_condition_specific_data.shape[0] == 0:
					log.debug('Skipping \'{}\' because there are no data meeting condition'.format(inconsistency_rule_name))
					continue

			# check if data has conflicts
			inconsistency_statement = inconsistency_rule.find('inconsistency')
			conflict_feature_name = inconsistency_statement.get('name') # Subject / Predicate / Object
			conflict_feature_values = [inconsistency_feature.get('value') for inconsistency_feature in inconsistency_statement]

			# skip to the next rule if there are no conflicts
			if self._data_has_conflict_values(pd_data[conflict_feature_name], conflict_feature_values) == False:
				log.debug('Skipping \'{}\' because there are no conflicts'.format(inconsistency_rule_name))
				continue

			rest_feature_names = [feature_name for feature_name in self._SPO_LIST if feature_name != conflict_feature_name]
			pd_grouped_data = pd_data.groupby(rest_feature_names)[conflict_feature_name].apply(set)

			def has_conflict_values(x, conflict_feature_values):
				return x.intersection(conflict_feature_values)

			pd_nconflict_data = pd_grouped_data.apply(has_conflict_values, args=(set(conflict_feature_values), ))
			pd_filtered_data = pd_nconflict_data[pd_nconflict_data.apply(len) > 1]

			# create inconsistency triple list
			for row_idx in range(pd_filtered_data.shape[0]):
				pd_conflict_data = pd.Series(pd_filtered_data.index[row_idx], index=rest_feature_names)

				conflict_tuples = []
				for conflict_value in pd_filtered_data[row_idx]:
					pd_conflict_data[conflict_feature_name] = conflict_value
					sources = pd.unique(pd_condition_specific_data[(pd_condition_specific_data[self._SPO_LIST] == pd_conflict_data).all(1)]['Source'])
					conflict_tuples.append((tuple(pd_conflict_data[self._SPO_LIST]), sources.tolist()))

				inconsistencies[inconsistency_id] = conflict_tuples
				inconsistency_id = inconsistency_id + 1

		log.info('Found {} inconsistencies'.format(len(inconsistencies)))

		return inconsistencies

	def _data_has_conflict_values(self, all_feature_values, conflict_feature_values):
		"""
		(Private) Check is data has conflicting values.

		Inputs:
			all_feature_values: column from one of Subject/Predicate/Object
			conflict_feature_values: list containing the conflicting feature values
				['response to antibiotic', 'no response to antibiotic']

		Returns:
			True if data has conflicting values, False otherwise.
		"""
		unique_feature_values  = pd.unique(all_feature_values)
		num_of_conflict_values = 0

		for conflict_feature_value in conflict_feature_values:
			if conflict_feature_value in unique_feature_values:
				num_of_conflict_values = num_of_conflict_values + 1

		return num_of_conflict_values > 1
