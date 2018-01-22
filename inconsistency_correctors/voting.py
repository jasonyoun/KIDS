import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

class Voting():
	@classmethod
	def resolve_inconsistencies(cls, data, inconsistencies):
		inconsistent_tuples_with_max_occurrence = []
		for inconsistent_tuples in inconsistencies:
			occurrences = Counter([inconsistent_tuple for inconsistent_tuple, sources in inconsistent_tuples])
			inconsistent_tuple, max_occurrence = max(occurrences.items(), key=operator.itemgetter(1))
			inconsistent_tuples_with_max_occurrence.append((inconsistent_tuple, max_occurrence))
		return inconsistent_tuples_with_max_occurrence, None