"""
Filename: utilities.py

Authors:
	Minseung Kim - msgkim@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
	Collection of utility functions.

To-do:
	1. clean-up and put comments
"""

import pandas as pd
import logging as log

def get_pd_of_statement(statement):
	"""
	Get names and values of the statement from data rules.

	Inputs:
		statement: XML node containing the statement to process

	Returns:
		All found values in series.
	"""
	feature_names   = [feature.get('name') for feature in statement]
	feature_values  = [feature.get('value') for feature in statement]

	return pd.Series(feature_values, index=feature_names)

def measure_accuracy(resolved_inconsistencies, answers, iteration=0):
	correctly_resolved_inconsistencies = 0.0
	total_attempted_resolution = 0.0

	for inconsistency_id in resolved_inconsistencies:
		resolved_inconsistency = resolved_inconsistencies[inconsistency_id]
		resolved_tuple  = resolved_inconsistency[0][0]
		resolved_belief = resolved_inconsistency[0][2]
		conflict_tuple  = resolved_inconsistency[1][0]
		conflict_belief = resolved_inconsistency[1][2]

		pd_resolved_inconsistency = pd.Series(resolved_tuple, index = SPO_LIST)
		pd_conflict_inconsistency = pd.Series(conflict_tuple, index = SPO_LIST)

		if (pd_resolved_inconsistency == answers[SPO_LIST]).all(1).any():
			correctly_resolved_inconsistencies = correctly_resolved_inconsistencies + 1
			total_attempted_resolution = total_attempted_resolution + 1
			log.debug('{}\tTRUE\t{}\t{}'.format(iteration, resolved_belief, conflict_belief))
		elif (pd_conflict_inconsistency == answers[SPO_LIST]).all(1).any():
			total_attempted_resolution = total_attempted_resolution + 1
			log.debug('{}\tFALSE\t{}\t{}'.format(iteration, resolved_belief, conflict_belief))

	log.debug('{} {}'.format(correctly_resolved_inconsistencies, total_attempted_resolution))
	accuracy = 0 if float(total_attempted_resolution) == 0 else float(correctly_resolved_inconsistencies) / float(total_attempted_resolution)
	return "{0:.4f}".format(accuracy)

def measure_trustworthiness(pd_data, answers):
	sources = pd.unique(pd_data['Source']).tolist()
	pd_trustworthiness = pd.Series(index = sources)

	for source in sources:
		source_claims = pd_data[pd_data['Source'] == source][SPO_LIST]
		common = source_claims.merge(answers,on=SPO_LIST)
		pd_trustworthiness[source] = float(common.shape[0]) / float(source_claims.shape[0])

	return pd_trustworthiness

def get_belief_of_inconsistencies(inconsistencies_with_max_belief, answer):
	data = {0: [], 1: []}

	for inconsistent_tuples_with_max_belief in inconsistencies_with_max_belief.values():
		belief = inconsistent_tuples_with_max_belief[0][2]

		pd_claim = pd.Series(inconsistent_tuples_with_max_belief[0][0], index = SPO_LIST)
		if (answer[SPO_LIST] == pd_claim).all(1).any():
			data[0] = data[0] + [belief]
		else:
			data[1] = data[1] + [belief]

	return data
