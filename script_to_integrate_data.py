"""
Filename: script_to_integrate_data.py

Authors:
	Minseung Kim - msgkim@ucdavis.edu
	Jason Youn -jyoun@ucdavis.edu

Description:
	Integrate the data.

To-do:
	1. check file existence and sanity check
	2. output intermediate and output files into a separate output folder
"""

#!/usr/bin/python

# import from generic packages
import sys
import argparse
import numpy as np
import pandas as pd
import logging as log

# import from knowledge_scholar package
from modules.data_manager import DataManager
from modules.inconsistency_manager import InconsistencyManager
from modules.report_manager import plot_trustworthiness, save_resolved_inconsistencies, save_integrated_data
from modules.inconsistency_correctors.averagelog import AverageLog

# default file paths
DATA_PATH_STR = './data/data_path_file.txt'
MAP_STR = './data/data_map.txt'
DATA_RULE_STR = './data/data_rules.xml'
INCONSISTENCY_RULES_STR = './data/inconsistency_rules.xml'
DATA_OUT_STR = 'out.txt'
INCONSISTENCY_OUT_STR = 'inconsistency.txt'

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
		default=DATA_PATH_STR,
		help='Path to the file data_path_file.txt')
	parser.add_argument(
		'--map',
		default=MAP_STR,
		help='Path to the file data_map.txt')
	parser.add_argument(
		'--data_rule',
		default=DATA_RULE_STR,
		help='Path to the file data_rules.xml')
	parser.add_argument(
		'--inconsistency_rule',
		default=INCONSISTENCY_RULES_STR,
		help='Path to the file inconsistency_rules.xml')
	parser.add_argument(
		'--data_out',
		default=DATA_OUT_STR,
		help='Path to save the integrated data file')
	parser.add_argument(
		'--inconsistency_out',
		default=INCONSISTENCY_OUT_STR,
		help='Path to save the inconsistencies file')

	return parser.parse_args()

if __name__ == '__main__':
	# set log and parse args
	set_logging()
	args = parse_argument()

	# perform 1) knowledge integration and 2) knowledge rule application
	pd_data = DataManager(args.data_path, args.map, args.data_rule).integrate_data()

	# perform inconsistency detection
	inconsistencies = InconsistencyManager(args.inconsistency_rule).detect_inconsistencies(pd_data)

	# resolve inconsistencies
	resolve_inconsistencies_result = AverageLog.resolve_inconsistencies(pd_data, inconsistencies)
	inconsistencies_with_max_belief = resolve_inconsistencies_result[0]
	pd_belief_and_source_without_inconsistencies = resolve_inconsistencies_result[1]
	np_trustworthiness_vector = resolve_inconsistencies_result[2]

	# report data integration results
	plot_trustworthiness(pd_data, np_trustworthiness_vector, inconsistencies)

	# save inconsistencies
	save_resolved_inconsistencies(args.inconsistency_out, inconsistencies_with_max_belief)

	# save integrated data
	save_integrated_data(args.data_out, pd_belief_and_source_without_inconsistencies)
