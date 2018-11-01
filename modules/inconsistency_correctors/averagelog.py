"""
Filename: averagelog.py

Authors:
	Minseung Kim - msgkim@ucdavis.edu

Description:
	Resolve inconsistencies using Average Log algorithm.

To-do:
	1. Change np.matrix into np.array for future compatibility.
"""

import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .sums import Sums
from ..utilities import measure_accuracy

MAX_NUM_ITERATIONS = 10
SPO_LIST = ['Subject','Predicate','Object']
THRESHOLD = np.power(0.1,10)

class AverageLog(object):
	@classmethod
	def resolve_inconsistencies(cls, pd_data, inconsistencies, answers=None):
		"""
		Resolve any inconsistency using Average Log algorithm.

		Inputs:
			pd_data: integrated data that needs inconsistencies resolved
			inconsistencies: Dictionary containing inconsistency_id as key
				and list of inconsistent triples + source as value

				{
				0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
					(('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
				1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
					('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
										...
				}
			answers: 

		Returns:
			inconsistencies_with_max_belief: dictionary where the key inconsistency_id
				and the value is a list of tuples where each tuple is of form
				(inconsistent_tuple, sources, belief). value of the dictionary is
				sorted by belief from high to low.
			pd_present_belief_and_source_without_inconsistencies: belief vector and
				pd_grouped_data concatenated but without the inconsistencies
			np_present_trustworthiness_vector: vector containing trustworthiness
				of all the sources
		"""
		# preprocess
		pd_source_size_data = pd_data.groupby('Source').size()
		pd_grouped_data = pd_data.groupby(SPO_LIST)['Source'].apply(set)

		# initialize
		np_present_belief_vector = Sums.initialize_belief(pd_grouped_data)
		np_past_trustworthiness_vector = None
		np_a_matrix, np_b_matrix = cls.create_matrices(pd_grouped_data, pd_source_size_data)

		delta = 1.0
		past_accuracy = 0.0
		iteration = 1

		# update until it reaches convergence
		while delta > THRESHOLD and iteration < MAX_NUM_ITERATIONS:
			# trustworthiness of all sources (10, 651626) x (651626, 1) = (10, 1)
			np_present_trustworthiness_vector = Sums.normalize(np_a_matrix.dot(np_present_belief_vector))
			# belief of all claims (651626, 10) x (10, 1) = (651626, 1)
			np_present_belief_vector = Sums.normalize(np_b_matrix.dot(np_present_trustworthiness_vector))
			delta = Sums.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
			np_past_trustworthiness_vector = np_present_trustworthiness_vector

			if answers is not None:
				inconsistencies_with_max_belief, pd_present_belief_and_source_without_inconsistencies = Sums.find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data)
				accuracy = measure_accuracy(inconsistencies_with_max_belief, answers)

				if past_accuracy == accuracy:
					print("[{}] accuracy saturation {} {} {}".format(cls.__name__, iteration, delta, accuracy))
				else:
					print("[{}] iteration, delta and accuracy : {} {} {}".format(cls.__name__, iteration, delta, accuracy))
				past_accuracy = accuracy
			else:
				print("[{}] iteration and delta : {} {}".format(cls.__name__, iteration, delta))

			# update iteration
			iteration = iteration + 1

		inconsistencies_with_max_belief, pd_present_belief_and_source_without_inconsistencies = Sums.find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data)

		return inconsistencies_with_max_belief, pd_present_belief_and_source_without_inconsistencies, np_present_trustworthiness_vector

	@staticmethod
	def create_matrices(pd_grouped_data, pd_source_size_data):
		"""
		Create matrices to be used to compute the trustworthiness and belief vectors.

		Inputs:
			pd_grouped_data: data grouped by 'pd_data.groupby(SPO_LIST)['Source'].apply(set)'
				with input shape (a, )
			pd_source_size_data: data grouped by 'pd_data.groupby('Source').size()'
				with input shape (b, ) where b is number of sources

		Returns:
			np_a_matrix: collection of union of one-hot vectors of shape (b, a)
				transform belief to trustworthiness
			np_b_matrix: transpose of np_a_matrix
				transform trustworthiness to belief
		"""
		sources = pd_source_size_data.index.tolist()
		pd_belief_source_matrix = pd_grouped_data.apply(Sums.create_source_vector, args = (sources,))
		np_belief_source_matrix = np.matrix(pd_belief_source_matrix.tolist())
		source_sizes = np.array(pd_source_size_data)

		np_a_matrix = (np_belief_source_matrix / (source_sizes / np.log(source_sizes))).T
		np_b_matrix = np_belief_source_matrix

		return np_a_matrix, np_b_matrix
