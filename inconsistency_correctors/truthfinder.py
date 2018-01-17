import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .sums import Sums
from .investment import Investment

MAX_NUM_ITERATIONS = 10

class TruthFinder():
	@classmethod
	def get_resolved_inconsistencies(cls, data, inconsistencies):
		tuple_to_belief_and_sources = Sums.initialize_beliefs(data)
		source_to_trustworthiness_and_size = cls.initialize_trustworthiness(data, inconsistencies)
		change = 1.0
		iteration = 1.0

		while change > np.power(0.1,10) and iteration < MAX_NUM_ITERATIONS:
			tuple_to_belief_and_sources = cls.measure_beliefs(data, source_to_trustworthiness_and_size, tuple_to_belief_and_sources, inconsistencies)
			source_to_new_trustworthiness_and_size = cls.measure_trustworthiness(data, tuple_to_belief_and_sources)
			change = Sums.measure_trustworthiness_change(source_to_trustworthiness_and_size, source_to_new_trustworthiness_and_size)
			print(str(iteration)+"\t"+str(change))
			iteration = iteration + 1
			source_to_trustworthiness_and_size = source_to_new_trustworthiness_and_size
		return Sums.find_tuple_with_max_belief(inconsistencies, tuple_to_belief_and_sources)

	@staticmethod
	def initialize_trustworthiness(data, inconsistencies):
		source_to_trustworthiness_and_size = {}
		unique_sources = pd.unique(data['Source'])
		for unique_source in unique_sources: # for each source
			trustworthiness = -np.log(0.1)
			unique_tuples_to_source = data[data['Source'] == unique_source][['Subject','Predicate','Object']].drop_duplicates()
			source_to_trustworthiness_and_size[unique_source] = (trustworthiness, len(unique_tuples_to_source))
		return source_to_trustworthiness_and_size

	@staticmethod
	def measure_trustworthiness(data, tuple_to_belief_and_sources):
		source_to_trustworthiness_and_size = {}
		unique_sources = pd.unique(data['Source'])
		for unique_source in unique_sources: # for each source
			trustworthiness = 0.0 
			unique_tuples_to_source = data[data['Source'] == unique_source][['Subject','Predicate','Object']].drop_duplicates()
			for idx, unique_tuple in unique_tuples_to_source.iterrows():
				trustworthiness = trustworthiness + tuple_to_belief_and_sources[tuple(unique_tuple.values)][0]
			trustworthiness = trustworthiness / float(len(unique_tuples_to_source))
			trustworthiness = - np.log(1 - trustworthiness)
			source_to_trustworthiness_and_size[unique_source] = (trustworthiness, len(unique_tuples_to_source))
		return source_to_trustworthiness_and_size

	@staticmethod
	def measure_beliefs(data, source_to_trustworthiness_and_size, tuple_to_belief_and_sources, inconsistencies):
		for tuple in tuple_to_belief_and_sources:
			belief, sources = tuple_to_belief_and_sources[tuple]
			new_belief = 0.0
			for source in sources:
				trustworthiness, size = source_to_trustworthiness_and_size[source]
				new_belief = new_belief + trustworthiness
			conflict_sources = TruthFinder.find_conflict_sources(data, tuple, inconsistencies)
			for conflict_source in conflict_sources:
				trustworthiness, size = source_to_trustworthiness_and_size[conflict_source]
				new_belief = new_belief - trustworthiness
			new_belief = 1 / (1 + np.exp(-0.3 * new_belief))
			tuple_to_belief_and_sources[tuple] = (new_belief, sources)
		# normalize
		return tuple_to_belief_and_sources

	@staticmethod
	def find_conflict_sources(data, tuple, inconsistencies):
		conflict_sources = []
		for inconsistent_tuples in inconsistencies:
			exclusive_tuples = Investment.get_exclusive_tuples(tuple, inconsistencies)
			for exclusive_tuple in exclusive_tuples:
				conflict_sources = conflict_sources + TruthFinder.get_sources_of_tuple(data, exclusive_tuple)
		return conflict_sources

	@staticmethod
	def get_sources_of_tuple(data, input_tuple):
		sources = data[(data['Subject'] == input_tuple[0]) & (data['Predicate'] == input_tuple[1]) & \
					(data['Object'] == input_tuple[2])]['Source']
		return sources.values.tolist()