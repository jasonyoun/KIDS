"""
Filename: data_manager.py

Authors:
	Minseung Kim - msgkim@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
	Python class to manage the dataset.
	Step 1. Perform name mapping.
	Step 2. Apply data rule to infer new data.

To-do:
	1. Maybe split integrate_data() into two functions name_map() and data_rule()
	2. Move _SPO_LIST to global file.
"""

#!/usr/bin/python

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from .utilities import get_pd_of_statement

class DataManager:
	"""
	Class for managing the data.
	"""

	#######################################
	# we probably need to get rid of this #
	#######################################
	SPO_LIST = ['Subject', 'Predicate', 'Object']

	def __init__(self, pd_data_paths, map_file, data_rule_file):
		"""
		Class constructor for DataManager.

		Inputs:
			pd_data_paths: path & source for each dataset
			map_file: data name mapping file name
			data_rule_file: data rule file name
		"""
		self.pd_data_paths = pd_data_paths
		self.map_file = map_file
		self.data_rule_file = data_rule_file

	def integrate_data(self):
		"""
		Integrate data from multiple sources and
		generate name mapped, data rule applied data.

		Returns:
			pd_integrated_data: resulting integrated data

				       Subject     Predicate Object Source
				0          lrp  no represses   fadD  hiTRN
				1          fur  no represses   yfeD  hiTRN
				2          fnr  no represses   ybhN  hiTRN
				3          crp  no represses   uxuR  hiTRN
								...
		"""
		list_integrated_data = []

		# iterate over each dataset and perform name mappipng
		for idx, row in self.pd_data_paths.iterrows():
			pd_data = pd.read_csv(row['Path'], '\t')
			pd_data = self._name_map_data(pd_data)
			pd_data['Source'] = row['Source']

			list_integrated_data.append(pd_data)

			print("[data integration] Added {} tuples from source {}.".format(pd_data.shape[0],row['Source']))

		# apply data rule
		pd_integrated_data = pd.concat(list_integrated_data)
		pd_integrated_data = self._apply_data_rule(pd_integrated_data)
		pd_integrated_data.index = range(pd_integrated_data.shape[0]) # update the index

		print("[data integration] Total of {} tuples are integrated.".format(pd_integrated_data.shape[0]))

		return pd_integrated_data

	def _name_map_data(self, pd_data):
		"""
		(Private) Perform name mapping given data from single source.

		Inputs:
			pd_data: all the data from single source

		Returns:
			pd_converted_data: name mapped data
		"""
		# open name mapping file
		with open(self.map_file) as f:
			next(f) # skip the header
			map_file_content = f.readlines()

		# store dictionary of the name mapping information
		dict_map = {}
		for map_line in map_file_content:
			key, value = map_line.strip('\n').split('\t')
			dict_map[key] = value

		def has_mapping_name(x, dict_map):
			# return original if both subject and object are already using correct name
			if (x['Subject'] in dict_map.values()) and (x['Object'] in dict_map.values()):
				return x

			new_x = x.copy()
			if (x['Subject'] in dict_map.values()) and (x['Object'] in dict_map):
				# map the object name
				new_x['Object']  = dict_map[x['Object']]
			elif (x['Object'] in dict_map.values()) and (x['Subject'] in dict_map):
				# map the subject name
				new_x['Subject'] = dict_map[x['Subject']]
			elif (x['Subject'] in dict_map) and (x['Object'] in dict_map):
				# map both subject and object naems
				new_x['Subject'] = dict_map[x['Subject']]
				new_x['Object']  = dict_map[x['Object']]
			else:
				###########################
				# do we do anything here? #
				###########################
				pass

			return new_x

		pd_converted_data = pd_data.apply(has_mapping_name, axis=1, args=(dict_map, ))

		########################################
		# put a log here about processing data #
		########################################

		return pd_converted_data

	def _apply_data_rule(self, pd_data):
		"""
		(Private) Apply data rule and infer new data.

		Inputs:
			pd_data: name mapped data integrated from all the sources

		Returns:
			pd_new_data: data with new inferred data added
		"""
		data_rules  = ET.parse(self.data_rule_file).getroot()
		pd_new_data = pd_data.copy()

		# iterate over each data rule
		for data_rule in data_rules:
			# find all the triples that meets the data rule's if statement
			if_statement = data_rule.find('if')
			pd_if_statement = get_pd_of_statement(if_statement)
			indices_meeting_if = (pd_data[pd_if_statement.index] == pd_if_statement).all(1).values
			pd_rule_specific_new_data = pd_data[indices_meeting_if].copy()

			if pd_rule_specific_new_data.shape[0] == 0:
				###########################
				# do we do anything here? #
				###########################
				continue

			# get the then statement that we are going to apply
			# and change it with the original (predicate, object, or both)
			then_statement    = data_rule.find('then')
			pd_then_statement = get_pd_of_statement(then_statement)
			pd_rule_specific_new_data[pd_then_statement.index] = pd_then_statement.tolist()

			# append the new data to the original
			pd_new_data = pd_new_data.append(pd_rule_specific_new_data)

		# drop the duplicates
		pd_new_data = pd_new_data.drop_duplicates()

		print("[data integration] {} new tuples are added based on the data rule.".format(pd_new_data.shape[0]-pd_data.shape[0]))

		return pd_new_data

	# def _get_source_size(self, pd_data):
	# 	"""
	# 	(Private) Get size of the source.

	# 	Inputs:
	# 		pd_data:
	# 	"""
	# 	return pd_data.groupby('Source').size()
