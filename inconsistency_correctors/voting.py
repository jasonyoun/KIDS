import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

class Voting():
	@classmethod
	def get_resolved_inconsistencies(cls, data, inconsistencies):
		tuples_with_max_occurrence = []
		for inconsistent_tuples in inconsistencies:
			occurrences = Counter(inconsistent_tuples)
			tuple_with_max_occurrence = max(occurrences.items(), key=operator.itemgetter(1))[0]
			tuples_with_max_occurrence.append(tuple_with_max_occurrence)
		return tuples_with_max_occurrence