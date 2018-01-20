import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .sums import Sums

MAX_NUM_ITERATIONS = 10

class Investment():
	@classmethod
	def resolve_inconsistencies(cls, data, inconsistencies):
		tuple_to_belief_and_sources = cls.initialize_beliefs(data, inconsistencies)
		source_to_trustworthiness_and_size = cls.initialize_trustworthiness(data, tuple_to_belief_and_sources)
		change = 1.0
		iteration = 1.0

		while change > np.power(0.1,10) and iteration < MAX_NUM_ITERATIONS:
			source_to_new_trustworthiness_and_size = cls.measure_trustworthiness(data, tuple_to_belief_and_sources, source_to_trustworthiness_and_size)
			change = Sums.measure_trustworthiness_change(source_to_trustworthiness_and_size, source_to_new_trustworthiness_and_size)
			print(str(iteration)+"\t"+str(change))
			iteration = iteration + 1
			source_to_trustworthiness_and_size = source_to_new_trustworthiness_and_size
			tuple_to_belief_and_sources = cls.measure_beliefs(source_to_trustworthiness_and_size, tuple_to_belief_and_sources)

		return Sums.find_tuple_with_max_belief(inconsistencies, tuple_to_belief_and_sources)

	@staticmethod
	def initialize_beliefs(data, inconsistencies):
		unique_tuples = data[['Subject','Predicate','Object']].drop_duplicates()
		tuple_to_belief_and_sources = {}
		for idx, unique_tuple in unique_tuples.iterrows():
			sources = data[(data['Subject'] == unique_tuple['Subject']) & (data['Predicate'] == unique_tuple['Predicate']) & \
				(data['Object'] == unique_tuple['Object'])]['Source']
			tuple = (unique_tuple['Subject'],unique_tuple['Predicate'],unique_tuple['Object'])

			total_size_of_all_sources = float(len(sources))
			exclusive_tuples = Investment.get_exclusive_tuples(tuple, inconsistencies)
			for exclusive_tuple in exclusive_tuples:
				exclusive_sources = data[(data['Subject'] == exclusive_tuple[0]) & (data['Predicate'] == exclusive_tuple[1]) & \
				(data['Object'] == exclusive_tuple[2])]['Source']
				total_size_of_all_sources = total_size_of_all_sources + float(len(exclusive_sources))
			
			belief = float(len(sources)) / total_size_of_all_sources
			tuple_to_belief_and_sources[tuple] = (belief,sources.values.tolist())
		return tuple_to_belief_and_sources

	@staticmethod
	def get_exclusive_tuples(tuple, inconsistencies):
		for inconsistent_tuples in inconsistencies:
			if tuple in inconsistent_tuples:
				return [inconsistent_tuple for inconsistent_tuple in inconsistent_tuples if tuple != inconsistent_tuple]
		return []

	@staticmethod
	def initialize_trustworthiness(data, tuple_to_belief_and_sources):
		source_to_trustworthiness_and_size = {}
		unique_sources = pd.unique(data['Source'])
		for unique_source in unique_sources: # for each source
			trustworthiness = 1.0 
			unique_tuples_to_source = data[data['Source'] == unique_source][['Subject','Predicate','Object']].drop_duplicates()
			source_to_trustworthiness_and_size[unique_source] = (trustworthiness, len(unique_tuples_to_source))
		# normalize
		return Sums.normalize_by_max(source_to_trustworthiness_and_size)

	@staticmethod
	def measure_trustworthiness(data, tuple_to_belief_and_sources, source_to_trustworthiness_and_size):
		source_to_new_trustworthiness_and_size = {}
		unique_sources = pd.unique(data['Source'])
		for unique_source in unique_sources: # for each source T(i)(s)
			trustworthiness = 0.0 
			unique_tuples_to_source = data[data['Source'] == unique_source][['Subject','Predicate','Object']].drop_duplicates()
			(prev_target_trustworthiness, target_size) = source_to_trustworthiness_and_size[unique_source] # T(i-1)(s)
			for idx, unique_tuple in unique_tuples_to_source.iterrows(): # for each claim c in source s
				(belief, sources) = tuple_to_belief_and_sources[tuple(unique_tuple.values)] # belief = B(i-1)(c)
				weight = 0.0
				for source in sources: # for each source r in Sc (sources of c)
					(source_trustworthiness, size) = source_to_trustworthiness_and_size[source]
					weight = weight + source_trustworthiness / float(size)
				trustworthiness = trustworthiness + belief * prev_target_trustworthiness / (target_size * weight)
			source_to_new_trustworthiness_and_size[unique_source] = (trustworthiness, target_size)
		# normalize
		return Sums.normalize_by_max(source_to_new_trustworthiness_and_size)

	@staticmethod
	def measure_beliefs(source_to_trustworthiness_and_size, tuple_to_belief_and_sources):
		for tuple in tuple_to_belief_and_sources:
			belief, sources = tuple_to_belief_and_sources[tuple]
			new_belief = 0.0
			for source in sources:
				(trustworthiness, size) = source_to_trustworthiness_and_size[source]
				new_belief = new_belief + trustworthiness / float(size)
			tuple_to_belief_and_sources[tuple] = (np.power(new_belief, 1.2), sources)
		# normalize
		return Sums.normalize_by_max(tuple_to_belief_and_sources)
