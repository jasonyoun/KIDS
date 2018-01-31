import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .sums import Sums
from ..inconsistency_manager import measure_accuracy

SPO_LIST           = ['Subject', 'Predicate', 'Object']
MAX_NUM_ITERATIONS = 10
THRESHOLD          = np.power(0.1,10)

class Investment(object):
   @classmethod
   def resolve_inconsistencies(cls, pd_data, inconsistencies, answers = None, exponent = 1.2):
   	  # preprocess
      pd_source_size_data = pd_data.groupby('Source').size()
      pd_grouped_data     = pd_data.groupby(SPO_LIST)['Source'].apply(set)

      # initialize
      np_present_belief_vector       = Sums.normalize(cls.initialize_belief(pd_source_size_data, pd_grouped_data, inconsistencies))
      np_past_trustworthiness_vector = cls.initialize_trustworthiness(pd_source_size_data)
      np_a_matrix, np_b_matrix       = cls.create_matrices(pd_grouped_data, pd_source_size_data)
      np_a_matrix                    = cls.update_a_matrix(np_a_matrix, np_past_trustworthiness_vector, pd_source_size_data)

      function_s  = np.vectorize(cls.function_s, otypes = [np.float])

      delta     = 1.0
      iteration = 1

      while delta > THRESHOLD and iteration < MAX_NUM_ITERATIONS:
         np_present_trustworthiness_vector = Sums.normalize(np_a_matrix.dot(np_present_belief_vector))
         np_present_belief_vector          = Sums.normalize(function_s(np_b_matrix.dot(np_present_trustworthiness_vector), exponent))
         delta = Sums.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
         np_a_matrix                    = cls.update_a_matrix(np_a_matrix, np_present_trustworthiness_vector, pd_source_size_data)
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
   def function_s(x, exponent):
      return np.power(x, exponent)

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
      for inconsistent_tuples in inconsistencies.values():
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