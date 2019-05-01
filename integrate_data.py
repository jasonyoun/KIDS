"""
Filename: integrate_data.py

Authors:
	Minseung Kim - msgkim@ucdavis.edu
	Jason Youn -jyoun@ucdavis.edu

Description:
	Integrate the data.

To-do:
	1. check file existence and sanity check
	2. output intermediate and output files into a separate output folder
	3. do more cleanup for report_manager.py and others
"""

#!/usr/bin/python

# import from generic packages
import sys
import argparse
import numpy as np
import pandas as pd
import logging as log

# import from knowledge_scholar package
from integrate_modules.data_manager import DataManager
from integrate_modules.inconsistency_manager import InconsistencyManager
from integrate_modules.report_manager import plot_trustworthiness
from integrate_modules.inconsistency_correctors.averagelog import AverageLog
from integrate_modules.inconsistency_correctors.inconsistency_corrector import InconsistencyCorrector

# default arguments
DEFAULT_DATA_PATH_STR = './data/data_path_file.txt'
DEFAULT_MAP_STR = './data/data_map.txt'
DEFAULT_DATA_RULE_STR = './data/data_rules.xml'
DEFAULT_INCONSISTENCY_RULES_STR = './data/inconsistency_rules.xml'
DEFAULT_DATA_OUT_STR = './output/out.txt'
DEFAULT_INCONSISTENCY_OUT_STR = './output/inconsistency.txt'

def set_logging():
	"""
	Configure logging.
	"""
	log.basicConfig(format='(%(levelname)s) %(filename)s: %(message)s', level=log.DEBUG)

	# set logging level to WARNING for matplotlib
	logger = log.getLogger('matplotlib')
	logger.setLevel(log.WARNING)

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Integrate knowledgebase from multiple sources.')

	parser.add_argument(
		'--data_path',
		default=DEFAULT_DATA_PATH_STR,
		help='Path to the file data_path_file.txt')
	parser.add_argument(
		'--map',
		default=DEFAULT_MAP_STR,
		help='Path to the file data_map.txt')
	parser.add_argument(
		'--data_rule',
		default=DEFAULT_DATA_RULE_STR,
		help='Path to the file data_rules.xml')
	parser.add_argument(
		'--inconsistency_rule',
		default=DEFAULT_INCONSISTENCY_RULES_STR,
		help='Path to the file inconsistency_rules.xml')
	parser.add_argument(
		'--data_out',
		default=DEFAULT_DATA_OUT_STR,
		help='Path to save the integrated data file')
	parser.add_argument(
		'--inconsistency_out',
		default=DEFAULT_INCONSISTENCY_OUT_STR,
		help='Path to save the inconsistencies file')
	parser.add_argument(
		'--use_temporal',
		default=False,
		action='store_true',
		help='Remove temporal data unless this option is set')

	return parser.parse_args()

if __name__ == '__main__':
	# set log and parse args
	set_logging()
	args = parse_argument()

	# perform 1) knowledge integration and 2) knowledge rule application
	dm = DataManager(args.data_path, args.map, args.data_rule)
	pd_data = dm.integrate_data()

	# remove temporal data in predicate
	if not args.use_temporal:
		pd_data = dm.drop_temporal_info(pd_data)

	# perform inconsistency detection
	im = InconsistencyManager(args.inconsistency_rule)
	inconsistencies = im.detect_inconsistencies(pd_data)

	ic = InconsistencyCorrector('AverageLog')
	resolve_inconsistencies_result = ic.resolve_inconsistencies(pd_data, inconsistencies)

	# resolve inconsistencies
	pd_resolved_inconsistencies = resolve_inconsistencies_result[0]
	pd_without_inconsistencies = resolve_inconsistencies_result[1]
	np_trustworthiness_vector = resolve_inconsistencies_result[2]


	# im.reinstate_resolved_inconsistencies(
	# 	pd_without_inconsistencies,
	# 	pd_resolved_inconsistencies)


	# report data integration results
	plot_trustworthiness(pd_data, np_trustworthiness_vector, inconsistencies)

	# save inconsistencies
	pd_resolved_inconsistencies.to_csv(args.inconsistency_out, index=False, sep='\t')

	# save integrated data
	pd_without_inconsistencies.to_csv(args.data_out, index=False, sep='\t')
