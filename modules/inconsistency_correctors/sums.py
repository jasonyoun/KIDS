"""
Filename: sums.py

Authors:
	Minseung Kim - msgkim@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
<<<<<<< HEAD
	Resolve inconsistencies Using Hubs and Authorities algorithm (a.k.a. Sums).
=======
	Resolve inconsistencies using Hubs and Authorities algorithm (a.k.a. Sums).
>>>>>>> inconsistency_manager

To-do:
	1. Change np.matrix into np.array for future compatibility.
	2. Check what answers do.
"""

import operator
import numpy as np
import pandas as pd
import math
from operator import itemgetter
from ..utilities import measure_accuracy

MAX_NUM_ITERATIONS = 10
SPO_LIST = ['Subject','Predicate','Object']
THRESHOLD = np.power(0.1,10)

class Sums(object):
	@classmethod
	def resolve_inconsistencies(cls, pd_data, inconsistencies, answers=None):
		"""
		Resolve any inconsistency using Sums algorithm.

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
		pd_source_size_data = pd_data.groupby('Source').size() # number of triples per each source
		pd_grouped_data = pd_data.groupby(SPO_LIST)['Source'].apply(set)

		# initialize
		np_present_belief_vector = cls.initialize_belief(pd_grouped_data) # (651626, 1) where all entries are 0.5
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
			delta = cls.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
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

		inconsistencies_with_max_belief, pd_present_belief_and_source_without_inconsistencies = cls.find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data)

		return inconsistencies_with_max_belief, pd_present_belief_and_source_without_inconsistencies, np_present_trustworthiness_vector

	@staticmethod
	def normalize(vector):
		"""
		Normalize a vector to prevent growing unbounded.

		Inputs:
			vector: vector to normalize

		Returns:
			normalized vector
		"""
		return vector / max(vector)

	@staticmethod
	def find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data):
		"""
		Given tuples of inconsistency, find one with maximum belief.

		Inputs:
			np_present_belief_vector: vector containing belief value of all the claims
			inconsistencies: Dictionary containing inconsistency_id as key
				and list of inconsistent triples + source as value

				{
				0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
					(('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
				1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
					('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
										...
				}
			pd_grouped_data: data grouped by 'pd_data.groupby(SPO_LIST)['Source'].apply(set)'
				with input shape (a, )

		Returns:
			inconsistencies_with_max_belief: dictionary where the key inconsistency_id
				and the value is a list of tuples where each tuple is of form
				(inconsistent_tuple, sources, belief). value of the dictionary is
				sorted by belief from high to low.
			pd_present_belief_and_source_without_inconsistencies: belief vector and
				pd_grouped_data concatenated but without the inconsistencies
		"""
		pd_present_belief_vector = pd.DataFrame(np_present_belief_vector, index=pd_grouped_data.index)
		pd_present_belief_and_source = pd.concat([pd_present_belief_vector, pd_grouped_data], axis=1)

		pd_present_belief_and_source_without_inconsistencies = pd_present_belief_and_source
		inconsistencies_with_max_belief = {}

		# loop through each inconsistency
		for inconsistency_id in inconsistencies:
			inconsistent_tuples = inconsistencies[inconsistency_id]
			inconsistent_tuples_with_max_belief = []

			# for each inconsistency id, loop through their contents
			for inconsistent_tuple, sources in inconsistent_tuples:
				belief = pd_present_belief_and_source.loc[inconsistent_tuple].values[0]
				# print('[inconsistency id {}] {} {}'.format(inconsistency_id,' '.join(inconsistent_tuple), belief))
				inconsistent_tuples_with_max_belief.append((inconsistent_tuple, sources, belief))
				pd_present_belief_and_source_without_inconsistencies = pd_present_belief_and_source_without_inconsistencies.drop(inconsistent_tuple)

			# sort inconsistencies by belief from high to low
			inconsistent_tuples_with_max_belief = sorted(inconsistent_tuples_with_max_belief, key=itemgetter(2), reverse=True)
			inconsistencies_with_max_belief[inconsistency_id] = inconsistent_tuples_with_max_belief

		return inconsistencies_with_max_belief, pd_present_belief_and_source_without_inconsistencies

	@staticmethod
	def measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector):
		"""
		Given previous iteration and current iteration trustworthiness vectors,
		calculate the mean absolute value.

		Inputs:
			np_past_trustworthiness_vector: previous iteration trustworthiness vector
			np_present_trustworthiness_vector: current iteration trustworthiness vector

		Returns:
			math.inf if i is 0, mean absolute value otherwise
		"""
		if np_past_trustworthiness_vector is None:
			return math.inf
		else:
			return np.mean(np.absolute(np_past_trustworthiness_vector - np_present_trustworthiness_vector))

	@staticmethod
	def initialize_belief(pd_grouped_data):
		"""
		Initialize the belief vector prior to 0.5.

		Inputs:
			pd_grouped_data: data grouped by 'pd_data.groupby(SPO_LIST)['Source'].apply(set)'
				with input shape (a, )

		Returns:
			numpy matrix of 0.5s of shape (a, 1)
		"""
		return np.matrix(pd_grouped_data.apply(lambda x: 0.5)).T

	@staticmethod
	def create_source_vector(elements, sources):
		"""
		Function to be called in order to create source vector.
		Given elements {'Liu', 'Tamae'} and sources
		['CARD', 'GO', 'Girgis', 'Liu', 'Nichols', 'Shaw', 'Soo', 'Tamae', 'Zhou', 'hiTRN']
		return the following output [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]

		Inputs:
			elements: sources that the SPO belongs to
			sources: list containing different sources

		Returns:
			union of one-hot vectors given different sources
		"""
		source_vector = np.zeros((len(sources), 1))

		return [1 if sources[idx] in list(elements) else 0 for idx in range(len(sources))]

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
<<<<<<< HEAD
			np_b_matrix: transpose of np_a_matrix
=======
				transform belief to trustworthiness
			np_b_matrix: transpose of np_a_matrix
				transform trustworthiness to belief
>>>>>>> inconsistency_manager
		"""
		sources = pd_source_size_data.index.tolist()
		pd_belief_source_matrix = pd_grouped_data.apply(Sums.create_source_vector, args=(sources,))
		np_belief_source_matrix = np.matrix(pd_belief_source_matrix.tolist())

		np_a_matrix = np_belief_source_matrix.T
		np_b_matrix = np_belief_source_matrix

		return np_a_matrix, np_b_matrix

	@staticmethod
	def get_source_size(pd_data, source):
		return pd_data[pd_data['Source'] == source].shape[0]
