"""
Filename: utilities.py

Authors:
	Minseung Kim - msgkim@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
	Collection of utility functions.
"""

import pandas as pd

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
