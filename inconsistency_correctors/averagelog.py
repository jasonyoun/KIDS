import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .sums       import Sums
from ..inconsistency_manager import measure_accuracy

MAX_NUM_ITERATIONS = 10
SPO_LIST           = ['Subject','Predicate','Object']
THRESHOLD          = np.power(0.1,10)

class AverageLog(object):
   @classmethod
   def resolve_inconsistencies(cls, pd_data, inconsistencies, answers = None):
      # preprocess
      pd_source_size_data = pd_data.groupby('Source').size()
      pd_grouped_data     = pd_data.groupby(SPO_LIST)['Source'].apply(set)

      # initialize
      np_present_belief_vector       = Sums.initialize_belief(pd_grouped_data)
      np_past_trustworthiness_vector = None
      np_a_matrix, np_b_matrix       = cls.create_matrices(pd_grouped_data, pd_source_size_data)

      delta     = 1.0
      iteration = 1

      while delta > THRESHOLD and iteration < MAX_NUM_ITERATIONS:
         np_present_trustworthiness_vector = Sums.normalize(np_a_matrix.dot(np_present_belief_vector))
         np_present_belief_vector          = Sums.normalize(np_b_matrix.dot(np_present_trustworthiness_vector))
         delta = Sums.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
         np_past_trustworthiness_vector = np_present_trustworthiness_vector

         if answers is not None:
            inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies = Sums.find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data)
            accuracy = measure_accuracy(inconsistencies_with_max_belief, answers)
            print("[{}] iteration, delta and accuracy : {} {} {}".format(cls.__name__, iteration, delta, accuracy))
         else:
            print("[{}] iteration and delta : {} {}".format(cls.__name__, iteration, delta))
         iteration = iteration + 1
      
      inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies = Sums.find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data)
      return inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies, np_present_trustworthiness_vector

   @staticmethod
   def create_matrices(pd_grouped_data, pd_source_size_data):
      sources                 = pd_source_size_data.index.tolist()
      pd_belief_source_matrix = pd_grouped_data.apply(Sums.create_source_vector, args = (sources,)) 
      np_belief_source_matrix = np.matrix(pd_belief_source_matrix.tolist()) 
      source_sizes            = np.array(pd_source_size_data)

      # np_a_matrix = transform trustworthiness to belief
      np_a_matrix = (np_belief_source_matrix / (source_sizes / np.log(source_sizes))).T
      np_b_matrix = np_belief_source_matrix

      return np_a_matrix, np_b_matrix