import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .sums       import Sums

MAX_NUM_ITERATIONS = 10
SPO_LIST = ['Subject','Predicate','Object']

class AverageLog(object):
   @classmethod
   def resolve_inconsistencies(cls, pd_data, inconsistencies):
      pd_present_belief_vector       = Sums.initialize_belief(pd_data)
      np_present_belief_vector       = np.matrix(pd_present_belief_vector)
      np_past_trustworthiness_vector = None
      np_a_matrix, np_b_matrix       = cls.create_matrices(pd_data)

      delta       = 1.0
      iteration   = 1

      while delta > np.power(0.1,10) and iteration < MAX_NUM_ITERATIONS:
         np_present_trustworthiness_vector = np_a_matrix.dot(np_present_belief_vector)
         np_present_belief_vector       = np_b_matrix.dot(np_present_trustworthiness_vector)
         delta = Sums.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
         np_past_trustworthiness_vector = np_present_trustworthiness_vector

         print("[truthfinder] iteration {} and delta {}".format(iteration, delta))
         iteration = iteration + 1
      
      pd_present_belief_vector     = pd.DataFrame(np_present_belief_vector, index = pd_present_belief_vector.index)
      pd_grouped_data              = pd_data.groupby(SPO_LIST)['Source'].apply(set)
      pd_present_belief_and_source = pd.concat([pd_present_belief_vector, pd_grouped_data], axis = 1)

      inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies = Sums.find_tuple_with_max_belief(inconsistencies, pd_present_belief_and_source)
      return inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies, np_present_trustworthiness_vector

   @staticmethod
   def create_matrices(pd_data):
      pd_grouped_data = pd_data.groupby(SPO_LIST)['Source'].apply(set)
      sources         = pd.unique(pd_data['Source']).tolist()

      pd_belief_source_matrix = pd_grouped_data.apply(Sums.create_source_vector, args = (sources,)) 
      np_belief_source_matrix = np.matrix(pd_belief_source_matrix.tolist()) 
      size_of_sources         = np.array([Sums.get_source_size(pd_data, source) for source in sources])
      
      # np_a_matrix = transform trustworthiness to belief
      # np_b_matrix = transform belief to trustworthiness
      np_a_matrix = (np_belief_source_matrix / (size_of_sources / np.log(size_of_sources))).T
      np_b_matrix = np_belief_source_matrix
      
      return np_a_matrix, np_b_matrix