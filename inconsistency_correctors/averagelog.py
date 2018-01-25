import operator
import numpy as np
import pandas as pd
import math
from collections import Counter
from .sums import Sums

MAX_NUM_ITERATIONS = 10

class AverageLog():
	@classmethod
	def resolve_inconsistencies(cls, data, inconsistencies):
		tuple_to_belief_and_sources = Sums.initialize_beliefs(data)
		source_to_trustworthiness_and_size = {}
		change = 1.0
		iteration = 1.0

		while change > np.power(0.1,10) and iteration < MAX_NUM_ITERATIONS:
			source_to_new_trustworthiness_and_size = cls.measure_trustworthiness(data, tuple_to_belief_and_sources)
			change = Sums.measure_trustworthiness_change(source_to_trustworthiness_and_size, source_to_new_trustworthiness_and_size)
			print(str(iteration)+"\t"+str(change))
			iteration = iteration + 1
			source_to_trustworthiness_and_size = source_to_new_trustworthiness_and_size
			tuple_to_belief_and_sources = Sums.measure_beliefs(source_to_trustworthiness_and_size, tuple_to_belief_and_sources)

		inconsistencies_with_max_belief, tuple_to_belief_and_sources_without_inconsistencies = Sums.find_tuple_with_max_belief(inconsistencies, tuple_to_belief_and_sources)
		return inconsistencies_with_max_belief, None, None

	@staticmethod
	def measure_trustworthiness(data, tuple_to_belief_and_sources):
		source_to_trustworthiness_and_size = {}
		unique_sources = pd.unique(data['Source'])
		for unique_source in unique_sources: # for each source
			trustworthiness = 0.0 
			unique_tuples_to_source = data[data['Source'] == unique_source][['Subject','Predicate','Object']].drop_duplicates()
			for idx, unique_tuple in unique_tuples_to_source.iterrows():
				trustworthiness = trustworthiness + tuple_to_belief_and_sources[tuple(unique_tuple.values)][0]
			trustworthiness = math.log(len(unique_tuples_to_source)) * (trustworthiness / float(len(unique_tuples_to_source)))
			source_to_trustworthiness_and_size[unique_source] = (trustworthiness, len(unique_tuples_to_source))
		# normalize
		return Sums.normalize_by_max(source_to_trustworthiness_and_size)
