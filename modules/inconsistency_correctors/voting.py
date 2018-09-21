import operator
import numpy as np
import pandas as pd
import math

class Voting():
	@classmethod
	def resolve_inconsistencies(cls, data, inconsistencies, answers = None):
		np_present_trustworthiness_vector = np.array(pd.Series(data.groupby('Source').size()))
		inconsistent_tuples_with_max_occurrence = {}

		for inconsistency_id in inconsistencies:
			inconsistent_tuples = inconsistencies[inconsistency_id]
			occurrences = {inconsistent_tuple: len(sources) for inconsistent_tuple, sources in inconsistent_tuples}
			inconsistent_tuple, max_occurrence = max(occurrences.items(), key=operator.itemgetter(1))
			inconsistent_tuples_with_max_occurrence[inconsistency_id] = [(inconsistent_tuple, max_occurrence), ('dummy',)]
		return inconsistent_tuples_with_max_occurrence, None, np_present_trustworthiness_vector