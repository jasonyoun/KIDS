# import generic packages
import operator
import numpy     as np
import pandas    as pd

# import knowledge_scholar packages
from .voting     import Voting
from .sums       import Sums
from .investment import Investment

# constants
MAX_NUM_ITERATIONS = 10
SPO_LIST           = ['Subject','Predicate','Object']

class TruthFinder():
   @classmethod
   def resolve_inconsistencies(cls, pd_data, inconsistencies):
      pd_present_belief_vector       = cls.initialize_belief(pd_data)
      np_present_belief_vector       = np.matrix(pd_present_belief_vector)
      np_past_trustworthiness_vector = cls.initialize_trustworthiness(pd_data)
      np_a_matrix, np_b_matrix       = cls.create_matrices(pd_data, inconsistencies)

      function_s  = np.vectorize(cls.function_s, otypes = [np.float])
      function_t  = np.vectorize(cls.function_t, otypes = [np.float])

      delta     = 1.0
      iteration = 1

      while delta > np.power(0.1,10) and iteration < MAX_NUM_ITERATIONS:
         np_present_belief_vector          = function_s(np_b_matrix.dot(np_past_trustworthiness_vector))
         np_present_trustworthiness_vector = function_t(np_a_matrix.dot(np_present_belief_vector))
         delta = Sums.measure_trustworthiness_change(np_past_trustworthiness_vector, np_present_trustworthiness_vector)
         np_past_trustworthiness_vector = np_present_trustworthiness_vector

         print("[{}] iteration {} and delta {}".format(cls.__name__, iteration, delta))
         iteration = iteration + 1
      
      pd_present_belief_vector     = pd.DataFrame(np_present_belief_vector, index = pd_present_belief_vector.index)
      pd_grouped_data              = pd_data.groupby(SPO_LIST)['Source'].apply(set)
      pd_present_belief_and_source = pd.concat([pd_present_belief_vector, pd_grouped_data], axis = 1)

      inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies = Sums.find_tuple_with_max_belief(inconsistencies, pd_present_belief_and_source)
      return inconsistencies_with_max_belief, pd_present_belief_vector_without_inconsistencies, np_present_trustworthiness_vector

   @staticmethod
   def initialize_trustworthiness(pd_data):
      num_of_sources            = len(pd.unique(pd_data['Source']))
      np_trustworthiness_vector = np.full((num_of_sources, 1), -np.log(0.1))

      return np_trustworthiness_vector

   @staticmethod
   def initialize_belief(pd_data):
      return pd.DataFrame(pd_data.groupby(SPO_LIST)['Source'].apply(lambda x: 0))

   @staticmethod
   def function_s(x):
      return 1 / (1 + np.exp(-0.3 * x))

   @staticmethod
   def function_t(x):
      return - np.log(1 - x)

   @staticmethod
   def modify_source_vector(elements, inconsistencies):
      for idx in range(len(elements)):
         if elements[idx] != 1 and TruthFinder.source_has_conflicting_belief(elements.index[idx], elements.name, inconsistencies):
            elements[idx] = -0.5
      return elements

   @staticmethod
   def create_matrices(pd_data, inconsistencies):
      pd_grouped_data = pd_data.groupby(SPO_LIST)['Source'].apply(set)
      sources         = pd.unique(pd_data['Source']).tolist()

      pd_belief_source_matrix = pd_grouped_data.apply(Sums.create_source_vector, args = (sources,)) 
      np_belief_source_matrix = np.matrix(pd_belief_source_matrix.tolist()) 
      size_of_sources         = np.array([Sums.get_source_size(pd_data, source) for source in sources])
      
      np_a_matrix = (np_belief_source_matrix / size_of_sources).T

      pd_belief_source_matrix = pd.DataFrame(np_belief_source_matrix, index = pd_grouped_data.index, columns = sources)
      np_b_matrix = pd_belief_source_matrix.apply(TruthFinder.modify_source_vector, args = (inconsistencies,)).as_matrix()
      
      return np_a_matrix, np_b_matrix

   @staticmethod
   def source_has_conflicting_belief(belief, source, inconsistencies):
      for inconsistent_tuples in inconsistencies:
         for (inconsistent_tuple, sources) in inconsistent_tuples:
            if tuple(belief) == inconsistent_tuple and source in sources:
               return True
      return False
