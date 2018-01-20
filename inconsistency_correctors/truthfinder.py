import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .voting import Voting
from .sums import Sums
from .investment import Investment

MAX_NUM_ITERATIONS = 10
SPO_LIST = ['Subject','Predicate','Object']

class TruthFinder():
   @classmethod
   def resolve_inconsistencies(cls, pd_data, inconsistencies):
      np_past_trustworthiness_vector = cls.initialize_trustworthiness(pd_data, inconsistencies)

      print("np_a_matrix")
      np_a_matrix = cls.create_a_matrix(pd_data)
      print("np_b_matrix")
      np_b_matrix = cls.create_b_matrix(pd_data, inconsistencies)

      function_s  = np.vectorize(cls.function_s, otypes=[np.float])
      function_t  = np.vectorize(cls.function_t, otypes=[np.float])

      delta       = 1.0
      iteration   = 1

      print('before while loop')
      while delta > np.power(0.1,10) and iteration < MAX_NUM_ITERATIONS:
         np_present_belief_vector          = function_s(np_b_matrix.dot(np_past_trustworthiness_vector))
         np_present_trustworthiness_vector = function_t(np_a_matrix.dot(np_present_belief_vector))
      
         delta = cls.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
         np_past_trustworthiness_vector = np_present_trustworthiness_vector 
         print("[truthfinder] iteration {} and delta {}".format(iteration, delta))
         iteration = iteration + 1

      return None, None

   @staticmethod
   def measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector):
      return np.mean(np.absolute(np_past_trustworthiness_vector - np_present_trustworthiness_vector))

   @staticmethod
   def function_s(x):
      return 1 / (1 + np.exp(-0.3 * x))

   @staticmethod
   def function_t(x):
      return - np.log(1 - x)

   @staticmethod
   def create_a_matrix(pd_data):
      beliefs     = pd_data[SPO_LIST].drop_duplicates()
      sources     = pd.unique(pd_data['Source'])
      pd_a_matrix = pd.DataFrame(np.zeros((len(sources), len(beliefs))), 
                                 index = sources.tolist(), 
                                 columns = [tuple(belief) for belief in beliefs.to_records(index=False)])

      list_num_of_sources = [TruthFinder.get_source_size(pd_data, source) for source in sources.tolist()]

      def mark_if_tuple_exist(elements, pd_data, list_num_of_sources):
         tuple = pd.Series(elements.name, index = SPO_LIST)
         found_sources  = pd_data[(pd_data[SPO_LIST] == tuple).values]['Source']
         
         return [1 / list_num_of_sources[idx] for idx in range(len(elements)) if elements.index[idx] in found_sources]

      pd_a_matrix = pd_a_matrix.apply(mark_if_tuple_exist, axis=0, args=(pd_data, list_num_of_sources))

      '''
      for row_idx in range(len(sources)):
         num_of_sources = TruthFinder.get_source_size(pd_data, sources[row_idx])
         pd_source      = pd.Series(sources[row_idx],index=['Source'])
         for col_idx in range(len(beliefs)):
            pd_belief_with_source = beliefs.iloc[col_idx].append(pd_source)
            if (pd_data == pd_belief_with_source).all(1).any():
               np_a_matrix[row_idx, col_idx] = 1 / num_of_sources
      '''

      return pd_a_matrix.as_matrix()

   @staticmethod
   def create_b_matrix(pd_data, inconsistencies):
      beliefs     = pd_data[SPO_LIST].drop_duplicates()
      sources     = pd.unique(pd_data['Source'])
      np_b_matrix = np.zeros((len(beliefs), len(sources)))
      
      for row_idx in range(len(beliefs)):
         for col_idx in range(len(sources)):
            pd_source             = pd.Series(sources[col_idx],index=['Source'])
            pd_belief_with_source = beliefs.iloc[row_idx].append(pd_source)
            if (pd_data == pd_belief_with_source).all(1).any():
               np_b_matrix[row_idx, col_idx] = 1
            elif TruthFinder.source_has_conflicting_belief(beliefs.iloc[row_idx], sources[col_idx], inconsistencies):
               np_b_matrix[row_idx, col_idx] = -1

      return np_b_matrix

   @staticmethod
   def source_has_conflicting_belief(belief, source, inconsistencies):
      for inconsistent_tuples in inconsistencies:
         for (inconsistent_tuple, sources) in inconsistent_tuples:
            if tuple(belief) == inconsistent_tuple and source in sources:
               return True
      return False

   @staticmethod
   def get_source_size(pd_data, source):
      return pd_data[pd_data['Source'] == source].shape[0]

   @staticmethod
   def initialize_trustworthiness(data, inconsistencies):
      num_of_sources = len(pd.unique(data['Source']))
      np_trustworthiness_vector = np.full((num_of_sources, 1), -np.log(0.1))

      return np_trustworthiness_vector
