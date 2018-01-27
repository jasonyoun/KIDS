import operator
import numpy as np
import pandas as pd
import math
from collections import Counter

from .sums import Sums

SPO_LIST           = ['Subject', 'Predicate', 'Object']
MAX_NUM_ITERATIONS = 10
THRESHOLD          = np.power(0.1,10)

class Investment():
	@classmethod
	def resolve_inconsistencies(cls, pd_data, inconsistencies):
		pd_claim_data  = cls.initialize_claim_data(pd_data, inconsistencies)
		pd_source_data = cls.initialize_source_data(pd_data)
		
		change    = 1.0
		iteration = 1.0

		while change > THRESHOLD and iteration < MAX_NUM_ITERATIONS:
			print(pd_source_data.head())
			pd_new_source_data = cls.measure_trustworthiness(pd_source_data, pd_claim_data)
			print('after measure_trustworthiness')
			print(pd_new_source_data.head())
			change = cls.measure_trustworthiness_change(pd_source_data, pd_new_source_data)
			print(str(iteration)+"\t"+str(change))
			iteration = iteration + 1
			pd_source_data = pd_new_source_data
			pd_claim_data = cls.measure_beliefs(pd_source_data, pd_claim_data)

		print(pd_claim_data.head())
		inconsistencies_with_max_belief, pd_consistent_claim_data = cls.find_tuple_with_max_belief(inconsistencies, pd_claim_data)
		return inconsistencies_with_max_belief, None, None

	@staticmethod
	def measure_trustworthiness_change(pd_source_data, pd_new_source_data):
		# TO BE IMPLEMENTED : DOESN'T AFFECT THE PERFORMANCE
		return math.inf

	@staticmethod
	def find_tuple_with_max_belief(inconsistencies, pd_claim_data):
		pd_consistent_claim_data = pd_claim_data.copy()
		inconsistent_tuples_with_max_belief = []
		inconsistency_idx = 1
		for inconsistent_tuples in inconsistencies:
			beliefs = {inconsistent_tuple: pd_claim_data.loc[[inconsistent_tuple],'Belief'].values[0] for inconsistent_tuple, sources in inconsistent_tuples}
			for tuple in beliefs:
				print('[inconsistency '+str(inconsistency_idx)+'] '+' '.join(tuple)+'\t'+str(beliefs[tuple]))
				#pd_consistent_claim_data.drop(tuple, None)
			inconsistency_idx = inconsistency_idx + 1
			inconsistent_tuple, max_belief = max(beliefs.items(), key=operator.itemgetter(1))
			inconsistent_tuples_with_max_belief.append((inconsistent_tuple, max_belief))

		return inconsistent_tuples_with_max_belief, pd_consistent_claim_data

	@staticmethod
	def initialize_claim_data(pd_data, inconsistencies):
		claims = pd_data[SPO_LIST].drop_duplicates()
		pd_claim_data = pd.DataFrame(index = claims, columns = ['Belief','Sources'])
		
		for idx, claim in claims.iterrows():
			sources = pd_data[(pd_data[SPO_LIST] == claim).all(1)]['Source']

			total_size_of_all_sources = float(len(sources))
			exclusive_claims          = Investment.get_exclusive_tuples(tuple(claim), inconsistencies)
			
			for exclusive_claim in exclusive_claims:
				pd_exclusive_claim        = pd.Series(exclusive_claim, index = SPO_LIST)
				exclusive_sources         = pd_data[(pd_data[SPO_LIST] == pd_exclusive_claim).all(1)]['Source']
				total_size_of_all_sources = total_size_of_all_sources + float(len(exclusive_sources))
		
			belief = float(len(sources)) / total_size_of_all_sources
			pd_claim_data.at[tuple(claim),'Belief']  = belief
			pd_claim_data.at[tuple(claim),'Sources'] = sources.values.tolist()
		
		return pd_claim_data

	@staticmethod
	def get_exclusive_tuples(tuple, inconsistencies):
		for inconsistent_tuples in inconsistencies:
			if tuple in [inconsistent_tuple for (inconsistent_tuple, sources) in inconsistent_tuples]:
				return [inconsistent_tuple for (inconsistent_tuple, sources) in inconsistent_tuples if tuple != inconsistent_tuple]
		return []

	@staticmethod
	def initialize_source_data(pd_data):
		pd_source_size_data     = pd_data.groupby('Source')[SPO_LIST].size()
		pd_claim_list_data      = pd.Series(dict(list(pd_data.groupby('Source')[SPO_LIST])))
		pd_trustworthiness_data = pd.Series(np.full(len(pd_source_size_data),1), index = pd_source_size_data.index)
		pd_source_data          = pd.concat([pd_source_size_data, pd_claim_list_data, pd_trustworthiness_data], axis = 1)
		pd_source_data.columns  = ['Size','Claims','Trustworthiness']
		return pd_source_data

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
