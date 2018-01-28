import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .investment import Investment
from .sums import Sums

SPO_LIST           = ['Subject', 'Predicate', 'Object']
MAX_NUM_ITERATIONS = 10
THRESHOLD          = np.power(0.1,10)

class PooledInvestment(object):
   @classmethod
   def resolve_inconsistencies(cls, pd_data, inconsistencies, exponent=1.4):
   	  # preprocess
      pd_source_size_data = pd_data.groupby('Source').size()
      pd_grouped_data     = pd_data.groupby(SPO_LIST)['Source'].apply(set)

      # initialize
      np_present_belief_vector       = Investment.initialize_belief(pd_source_size_data, pd_grouped_data, inconsistencies)
      np_past_trustworthiness_vector = Investment.initialize_trustworthiness(pd_source_size_data)
      np_a_matrix, np_b_matrix       = Investment.create_matrices(pd_grouped_data, pd_source_size_data)
      np_a_matrix                    = Investment.update_a_matrix(np_a_matrix, np_past_trustworthiness_vector, pd_source_size_data)

      delta     = 1.0
      iteration = 1

      while delta > THRESHOLD and iteration < MAX_NUM_ITERATIONS:
         np_present_trustworthiness_vector = np_a_matrix.dot(np_present_belief_vector)
         claims                            = pd_grouped_data.index.tolist()
         np_present_belief_vector          = cls.normalize(np_b_matrix.dot(np_present_trustworthiness_vector), claims, inconsistencies, exponent)
         delta = Sums.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
         np_a_matrix                    = Investment.update_a_matrix(np_a_matrix, np_present_trustworthiness_vector, pd_source_size_data)
         np_past_trustworthiness_vector = np_present_trustworthiness_vector

         print("[{}] iteration {} and delta {}".format(cls.__name__, iteration, delta))
         iteration = iteration + 1
      
      pd_present_belief_vector     = pd.DataFrame(np_present_belief_vector, index = pd_grouped_data.index)
      pd_present_belief_and_source = pd.concat([pd_present_belief_vector, pd_grouped_data], axis = 1)

      inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies = Sums.find_tuple_with_max_belief(inconsistencies, pd_present_belief_and_source)
      return inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies, np_present_trustworthiness_vector

   @staticmethod
   def normalize(np_present_belief_vector, claims, inconsistencies, exponent):
      np_new_belief_vector = np_present_belief_vector.copy()

      for inconsistent_tuples in inconsistencies.values():
         total_score = 0
         for (inconsistent_tuple, sources) in inconsistent_tuples:
            total_score = total_score + Investment.function_s(np_present_belief_vector[claims.index(inconsistent_tuple)], exponent)
         for (inconsistent_tuple, sources) in inconsistent_tuples:
            present_value         = np_new_belief_vector[claims.index(inconsistent_tuple)]
            claim_spepcific_value = Investment.function_s(np_present_belief_vector[claims.index(inconsistent_tuple)], exponent)
            np_new_belief_vector[claims.index(inconsistent_tuple)] = present_value * claim_spepcific_value / total_score
      
      return np_new_belief_vector
