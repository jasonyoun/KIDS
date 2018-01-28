import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .sums import Sums

SPO_LIST           = ['Subject', 'Predicate', 'Object']
MAX_NUM_ITERATIONS = 10
THRESHOLD          = np.power(0.1,10)

class Investment(object):
   @classmethod
   def resolve_inconsistencies(cls, pd_data, inconsistencies):
   	  # preprocess
      pd_source_size_data = pd_data.groupby('Source').size()
      pd_grouped_data     = pd_data.groupby(SPO_LIST)['Source'].apply(set)

      # initialize
      np_present_belief_vector       = cls.initialize_belief(pd_source_size_data, pd_grouped_data, inconsistencies)
      np_past_trustworthiness_vector = cls.initialize_trustworthiness(pd_source_size_data)
      np_a_matrix, np_b_matrix       = cls.create_matrices(pd_grouped_data, pd_source_size_data)
      np_a_matrix                    = cls.update_a_matrix(np_a_matrix, np_past_trustworthiness_vector, pd_source_size_data)

      function_s  = np.vectorize(cls.function_s, otypes = [np.float])

      delta       = 1.0
      iteration   = 1

      while delta > np.power(0.1,10) and iteration < MAX_NUM_ITERATIONS:
         np_present_trustworthiness_vector = np_a_matrix.dot(np_present_belief_vector)
         np_present_belief_vector          = function_s(np_b_matrix.dot(np_present_trustworthiness_vector))
         delta = Sums.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
         np_a_matrix                    = cls.update_a_matrix(np_a_matrix, np_present_trustworthiness_vector, pd_source_size_data)
         np_past_trustworthiness_vector = np_present_trustworthiness_vector

         print("[truthfinder] iteration {} and delta {}".format(iteration, delta))
         iteration = iteration + 1
      
      pd_present_belief_vector     = pd.DataFrame(np_present_belief_vector, index = pd_grouped_data.index)
      pd_present_belief_and_source = pd.concat([pd_present_belief_vector, pd_grouped_data], axis = 1)

      inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies = Sums.find_tuple_with_max_belief(inconsistencies, pd_present_belief_and_source)
      return inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies, np_present_trustworthiness_vector

   @staticmethod
   def function_s(x):
      return np.power(x, 1.2)

   @staticmethod
   def create_matrices(pd_grouped_data, pd_source_size_data):
      sources = pd_source_size_data.index.tolist()
      pd_belief_source_matrix = pd_grouped_data.apply(Sums.create_source_vector, args = (sources,)) 
      np_belief_source_matrix = np.matrix(pd_belief_source_matrix.tolist()) 

      # np_a_matrix = transform trustworthiness to belief
      np_a_matrix = np_belief_source_matrix.T
      np_b_matrix = np_belief_source_matrix / np.array(pd_source_size_data)
     
      return np_a_matrix, np_b_matrix

   @staticmethod
   def update_a_matrix(np_a_matrix, np_past_trustworthiness_vector, pd_source_size_data):
      np_a_matrix = (np_a_matrix.T / (np.array(pd_source_size_data) / np_past_trustworthiness_vector.T)).T
      return np_a_matrix / np_a_matrix.sum(axis = 1)

   @staticmethod
   def initialize_belief(pd_source_size_data, pd_grouped_data, inconsistencies):
      pd_present_belief_vector = pd_grouped_data.apply(lambda x: 1)
      # we only need to change claim that has inconsistency.
      for inconsistent_tuples in inconsistencies:
         total_source_size = Investment.get_total_source_size_of_inconsistent_tuples(inconsistent_tuples, pd_source_size_data)
         for (inconsistent_tuple, sources) in inconsistent_tuples:
            source_size = sum([pd_source_size_data[source] for source in sources])
            pd_present_belief_vector.loc[inconsistent_tuple] = float(source_size) / float(total_source_size)

      return np.matrix(pd_present_belief_vector).T

   @staticmethod
   def get_total_source_size_of_inconsistent_tuples(inconsistent_tuples, pd_source_size_data):
      total_source_size = 0
      for (inconsistent_tuple, sources) in inconsistent_tuples:
         for source in sources:
            total_source_size = total_source_size + pd_source_size_data[source]
      return total_source_size
 
   @staticmethod
   def initialize_trustworthiness(pd_source_size_data):
      return np.matrix(pd_source_size_data.apply(lambda x: 1)).T

'''
	@staticmethod
	def get_exclusive_tuples(tuple, inconsistencies):
		for inconsistent_tuples in inconsistencies:
			if tuple in [inconsistent_tuple for (inconsistent_tuple, sources) in inconsistent_tuples]:
				return [inconsistent_tuple for (inconsistent_tuple, sources) in inconsistent_tuples if tuple != inconsistent_tuple]
		return []

	@staticmethod
	def measure_trustworthiness(pd_source_data, pd_claim_data):
		pd_new_source_data = pd_source_data.copy()

		for target_source in pd_new_source_data.index: # for each source T(i)(s)
			new_trustworthiness = 0.0 
			(target_size, target_claims, target_trustworthiness) = pd_source_data.loc[target_source] # T(i-1)(s)
			for idx, target_claim in target_claims[SPO_LIST].iterrows(): # for each claim c in source s
				belief  = pd_claim_data.loc[[tuple(target_claim)],'Belief'].values[0] # belief = B(i-1)(c)
				sources = pd_claim_data.loc[[tuple(target_claim)],'Sources']
				weight  = 0.0
				
				for source in list(sources)[0]: # IT IS UGLY. NEED TO MODIFY
					(source_size, source_claims, source_trustworthiness) = pd_source_data.loc[source]
					weight = weight + source_trustworthiness / float(source_size)
				new_trustworthiness = new_trustworthiness + belief * target_trustworthiness / (target_size * weight)
			pd_new_source_data.at[[target_source],'Trustworthiness'] = new_trustworthiness
		# normalize
		return pd_new_source_data

	@staticmethod
	def measure_beliefs(pd_source_data, pd_claim_data):
		for claim in pd_claim_data.index:
			belief  = pd_claim_data.loc[[tuple(claim)],'Belief'].values[0]
			sources = pd_claim_data.loc[[tuple(claim)],'Sources']
			new_belief = 0.0
			for source in list(sources)[0]: # IT IS UGLY. NEED TO MODIFY
				(source_size, source_claims, source_trustworthiness) = pd_source_data.loc[source]
				new_belief = new_belief + source_trustworthiness / float(source_size)
			pd_claim_data.at[[claim],'Belief'] = np.power(new_belief, 1.2)
		# normalize
		return pd_claim_data
'''