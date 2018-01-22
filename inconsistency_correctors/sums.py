import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

MAX_NUM_ITERATIONS = 10
SPO_LIST = ['Subject','Predicate','Object']

class Sums(object):
	@classmethod
	def resolve_inconsistencies(cls, data, inconsistencies):
		tuple_to_belief_and_sources = cls.initialize_beliefs(data)
		source_to_trustworthiness_and_size = {}
		change = 1.0
		iteration = 1.0

		while change > np.power(0.1,10) and iteration < MAX_NUM_ITERATIONS:
			source_to_new_trustworthiness_and_size = cls.measure_trustworthiness(data, tuple_to_belief_and_sources)
			change = cls.measure_trustworthiness_change(source_to_trustworthiness_and_size, source_to_new_trustworthiness_and_size)
			print(str(iteration)+"\t"+str(change))
			iteration = iteration + 1
			source_to_trustworthiness_and_size = source_to_new_trustworthiness_and_size
			tuple_to_belief_and_sources = cls.measure_beliefs(source_to_trustworthiness_and_size, tuple_to_belief_and_sources)

		return cls.find_tuple_with_max_belief(inconsistencies, tuple_to_belief_and_sources)

	@staticmethod
	def tuple_is_inconsistent(tuple, inconsistencies):
		for inconsistent_tuples in inconsistencies:
			if tuple in inconsistent_tuples:
				return True
		return False

	@staticmethod
	def find_tuple_with_max_belief(inconsistencies, tuple_to_belief_and_sources):
		tuple_to_belief_and_sources_without_inconsistencies = tuple_to_belief_and_sources
		inconsistent_tuples_with_max_belief = []
		inconsistency_idx = 1
		for inconsistent_tuples in inconsistencies:
			beliefs = {inconsistent_tuple: tuple_to_belief_and_sources[inconsistent_tuple][0] for inconsistent_tuple, sources in inconsistent_tuples}
			for tuple in beliefs:
				print('[inconsistency '+str(inconsistency_idx)+'] '+' '.join(tuple)+'\t'+str(beliefs[tuple]))
				tuple_to_belief_and_sources_without_inconsistencies.pop(tuple, None)
			inconsistency_idx = inconsistency_idx + 1
			inconsistent_tuple, max_belief = max(beliefs.items(), key=operator.itemgetter(1))
			inconsistent_tuples_with_max_belief.append((inconsistent_tuple, max_belief))

		return inconsistent_tuples_with_max_belief, tuple_to_belief_and_sources_without_inconsistencies

	@staticmethod
	def initialize_beliefs(data):
		unique_tuples = data[SPO_LIST].drop_duplicates()
		tuple_to_belief_and_sources = {}
		for idx, unique_tuple in unique_tuples.iterrows():
			sources = data[(data['Subject'] == unique_tuple['Subject']) & (data['Predicate'] == unique_tuple['Predicate']) & \
				(data['Object'] == unique_tuple['Object'])]['Source']
			tuple = (unique_tuple['Subject'],unique_tuple['Predicate'],unique_tuple['Object'])
			tuple_to_belief_and_sources[tuple] = (0.5,sources.values.tolist())
		return tuple_to_belief_and_sources

	@staticmethod
	def measure_trustworthiness_change(source_to_trustworthiness_and_size, source_to_new_trustworthiness_and_size):
		if source_to_trustworthiness_and_size == {}:
			return math.inf
		differences = [np.absolute(source_to_trustworthiness_and_size[source][0] - source_to_new_trustworthiness_and_size[source][0]) for source in source_to_trustworthiness_and_size]
		return np.mean(differences)

	@staticmethod
	def measure_trustworthiness(data, tuple_to_belief_and_sources):
		source_to_trustworthiness_and_size = {}
		unique_sources = pd.unique(data['Source'])
		for unique_source in unique_sources: # for each source
			trustworthiness = 0.0 
			unique_tuples_to_source = data[data['Source'] == unique_source][SPO_LIST].drop_duplicates()
			for idx, unique_tuple in unique_tuples_to_source.iterrows():
				trustworthiness = trustworthiness + tuple_to_belief_and_sources[tuple(unique_tuple.values)][0]
			source_to_trustworthiness_and_size[unique_source] = (trustworthiness, len(unique_tuples_to_source))
		# normalize
		return Sums.normalize_by_max(source_to_trustworthiness_and_size)

	@staticmethod
	def measure_beliefs(source_to_trustworthiness_and_size, tuple_to_belief_and_sources):
		for tuple in tuple_to_belief_and_sources:
			belief, sources = tuple_to_belief_and_sources[tuple]
			new_belief = 0.0
			for source in sources:
				new_belief = new_belief + source_to_trustworthiness_and_size[source][0]
			tuple_to_belief_and_sources[tuple] = (new_belief, sources)
		# normalize
		return Sums.normalize_by_max(tuple_to_belief_and_sources)

	@staticmethod
	def normalize_by_max(dic):
		max_value = max(dic.items(), key=operator.itemgetter(1))[1][0]
		return {key: (value / max_value, rest) for key, (value, rest) in dic.items()}
