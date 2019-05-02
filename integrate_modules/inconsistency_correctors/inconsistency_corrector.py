"""
Filename: inconsistency_corrector.py

Authors:
	Jason Youn - jyoun@ucdavis.edu

Description:
	Correct inconsistencies using different methods.

To-do:
	1. comments
"""


import sys
import numpy as np
import pandas as pd
import logging as log
from ast import literal_eval
from .averagelog import AverageLog
from .investment import Investment
from .pooledinvestment import PooledInvestment
from .sums import Sums
from .truthfinder import TruthFinder
from .voting import Voting

class InconsistencyCorrector(AverageLog, Investment, PooledInvestment, Sums, TruthFinder, Voting):
	def __init__(self, mode):
		self.mode = mode.lower()

	def resolve_inconsistencies(self, pd_data, inconsistencies, **kwargs):
		if self.mode == 'averagelog':
			result = AverageLog.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
		elif self.mode == 'investment':
			result = Investment.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
		elif self.mode == 'pooledinvestment':
			result = PooledInvestment.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
		elif self.mode == 'sums':
			result = Sums.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
		elif self.mode == 'truthfinder':
			result = TruthFinder.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
		elif self.mode == 'voting':
			result = Voting.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
		else:
			log.error('Invalid inconsistency corrector \'{}\'chosen.'.format(self.mode))
			sys.exit()

		pd_resolved_inconsistencies = self._inconsistencies_dict_to_pd(result[0])
		pd_without_inconsistencies = self._without_inconsistencies_reformat(result[1])
		np_trustworthiness_vector = result[2]

		return (pd_resolved_inconsistencies, pd_without_inconsistencies, np_trustworthiness_vector)

	def _without_inconsistencies_reformat(self, pd_belief_and_source_without_inconsistencies):
		log.info('Reformating data without inconsistencies.')

		pd_without_inconsistencies = pd_belief_and_source_without_inconsistencies.reset_index()
		pd_without_inconsistencies = pd_without_inconsistencies.rename(columns={0: 'Belief'}).astype('str')

		pd_without_inconsistencies['Belief'] = pd_without_inconsistencies['Belief'].apply(lambda x: '{0:.2f}'.format(float(x)))
		pd_without_inconsistencies['Source size'] = pd_without_inconsistencies['Source'].apply(lambda x: len(literal_eval(x)))
		pd_without_inconsistencies['Sources'] = pd_without_inconsistencies['Source'].apply(lambda x: ','.join(literal_eval(x)))

		pd_without_inconsistencies.drop(columns='Source', inplace=True)

		return pd_without_inconsistencies

	def _inconsistencies_dict_to_pd(self, resolved_inconsistencies_dict):
		log.info('Converting resolved inconsistencies from dictionary to pandas DataFrame.')

		pd_resolved_inconsistencies = pd.DataFrame(columns=[
			'Subject', 'Predicate', 'Object', 'Belief', 'Source size', 'Sources',
			'Total source size', 'Mean belief of conflicting tuples', 'Belief difference',
			'Conflicting tuple info'])

		for inconsistent_tuples_with_max_belief in resolved_inconsistencies_dict.values():
			(selected_tuple, sources, belief) = inconsistent_tuples_with_max_belief[0]
			conflicting_tuple_info = inconsistent_tuples_with_max_belief[1:]

			total_source_size = np.sum([len(inconsistent_tuple_with_max_belief[1]) for inconsistent_tuple_with_max_belief in inconsistent_tuples_with_max_belief])
			mean_belief_of_conflicting_tuple = np.mean([_tuple_[2] for _tuple_ in conflicting_tuple_info])

			row_dict = {}
			row_dict['Subject'] = selected_tuple[0]
			row_dict['Predicate'] = selected_tuple[1]
			row_dict['Object'] = selected_tuple[2]
			row_dict['Belief'] = '{0:.10f}'.format(belief)
			row_dict['Source size'] = len(sources)
			row_dict['Sources'] = ','.join(sources)
			row_dict['Total source size'] = total_source_size
			row_dict['Mean belief of conflicting tuples'] = '{0:.10f}'.format(mean_belief_of_conflicting_tuple)
			row_dict['Belief difference'] = '{0:.10f}'.format(belief - mean_belief_of_conflicting_tuple)
			row_dict['Conflicting tuple info'] = str(conflicting_tuple_info)

			pd_resolved_inconsistencies = pd_resolved_inconsistencies.append(
				pd.DataFrame.from_records([row_dict]), ignore_index=True, sort=False)

		return pd_resolved_inconsistencies
