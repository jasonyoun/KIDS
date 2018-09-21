import operator
import numpy as np
import pandas as pd
import math
from operator import itemgetter
from ..inconsistency_manager import measure_accuracy

MAX_NUM_ITERATIONS = 10
SPO_LIST = ['Subject','Predicate','Object']
THRESHOLD          = np.power(0.1,10)

class Sums(object):
   @classmethod
   def resolve_inconsistencies(cls, pd_data, inconsistencies, answers = None):
      # preprocess
      pd_source_size_data = pd_data.groupby('Source').size()
      pd_grouped_data     = pd_data.groupby(SPO_LIST)['Source'].apply(set)

      # initialize
      np_present_belief_vector       = cls.initialize_belief(pd_grouped_data)
      np_past_trustworthiness_vector = None
      np_a_matrix, np_b_matrix       = cls.create_matrices(pd_grouped_data, pd_source_size_data)

      delta         = 1.0
      past_accuracy = 0.0
      iteration     = 1

      # update until it reaches convergence
      while delta > THRESHOLD and iteration < MAX_NUM_ITERATIONS:
         np_present_trustworthiness_vector = Sums.normalize(np_a_matrix.dot(np_present_belief_vector))
         np_present_belief_vector          = Sums.normalize(np_b_matrix.dot(np_present_trustworthiness_vector))
         delta = cls.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
         np_past_trustworthiness_vector = np_present_trustworthiness_vector

         if answers is not None:
            inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies = Sums.find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data)
            accuracy = measure_accuracy(inconsistencies_with_max_belief, answers)
            
            if past_accuracy == accuracy:
               print("[{}] accuracy saturation {} {} {}".format(cls.__name__, iteration, delta, accuracy))
            else:
               print("[{}] iteration, delta and accuracy : {} {} {}".format(cls.__name__, iteration, delta, accuracy))
            past_accuracy = accuracy
         else:
            print("[{}] iteration and delta : {} {}".format(cls.__name__, iteration, delta))
         iteration = iteration + 1
      
      inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies = cls.find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data)
      return inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies, np_present_trustworthiness_vector

   @staticmethod
   def normalize(vector):
      return vector / max(vector)

   @staticmethod
   def find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data):
      pd_present_belief_vector     = pd.DataFrame(np_present_belief_vector, index = pd_grouped_data.index)
      pd_present_belief_and_source = pd.concat([pd_present_belief_vector, pd_grouped_data], axis = 1)

      pd_present_belief_vector_without_inconsistencies = pd_present_belief_and_source
      inconsistencies_with_max_belief = {}
 
      for inconsistency_id in inconsistencies:
         inconsistent_tuples = inconsistencies[inconsistency_id]
         inconsistent_tuples_with_max_belief = []
         for inconsistent_tuple, sources in inconsistent_tuples:
            belief = pd_present_belief_and_source.loc[inconsistent_tuple].values[0]
            #print('[inconsistency id {}] {} {}'.format(inconsistency_id,' '.join(inconsistent_tuple), belief))
            inconsistent_tuples_with_max_belief.append((inconsistent_tuple, sources, belief))
            pd_present_belief_vector_without_inconsistencies = pd_present_belief_vector_without_inconsistencies.drop(inconsistent_tuple)

         inconsistent_tuples_with_max_belief = sorted(inconsistent_tuples_with_max_belief, key = itemgetter(2), reverse = True)
         inconsistencies_with_max_belief[inconsistency_id] = inconsistent_tuples_with_max_belief
 
      return inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies

   @staticmethod
   def measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector):
      if np_past_trustworthiness_vector is None:
         return math.inf
      else:    
         return np.mean(np.absolute(np_past_trustworthiness_vector - np_present_trustworthiness_vector))

   @staticmethod
   def initialize_belief(pd_grouped_data):
      return np.matrix(pd_grouped_data.apply(lambda x: 0.5)).T

   @staticmethod
   def create_source_vector(elements, sources):
      source_vector = np.zeros((len(sources), 1))
      return [1 if sources[idx] in list(elements) else 0 for idx in range(len(sources))]

   @staticmethod
   def create_matrices(pd_grouped_data, pd_source_size_data):
      sources = pd_source_size_data.index.tolist()
      pd_belief_source_matrix = pd_grouped_data.apply(Sums.create_source_vector, args = (sources,)) 
      np_belief_source_matrix = np.matrix(pd_belief_source_matrix.tolist())    
      np_a_matrix             = np_belief_source_matrix.T
      np_b_matrix             = np_belief_source_matrix
      
      return np_a_matrix, np_b_matrix

   @staticmethod
   def get_source_size(pd_data, source):
      return pd_data[pd_data['Source'] == source].shape[0]