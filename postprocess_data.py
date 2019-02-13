"""
Filename: postprocess_data.py

Authors:
	Jason Youn -jyoun@ucdavis.edu

Description:
	Perform postprocessing on the integrated data so that
	it's suitable for use in hypothesis generator.

To-do:
	1. make a general code which can be applied to multiple datasets
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import logging as log

from postprocess_modules.data_processor import DataProcessor
from postprocess_modules.extract_info import ExtractInfo
from postprocess_modules.split_folds import SplitFolds

# default file paths
DEFAULT_OUTPUT_PATH_STR = './output'
DEFAULT_DATA_PATH_STR = './output/out.txt'
DEFAULT_DR_PATH_STR = './data/domain_range.txt'

# default file names
DEFAULT_ENTITIES_TXT_STR = 'entities.txt'
DEFAULT_ENTITY_FULL_NAMES_TXT_STR = 'entity_full_names.txt'
DEFAULT_ALL_DATA_TXT_STR = 'data.txt'

# number of folds to split the dataset into
NUM_FOLDS = 5

def set_logging():
	"""
	Configure logging.
	"""
	log.basicConfig(format='(%(levelname)s) %(filename)s: %(message)s', level=log.DEBUG)

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Postprocess the integrated data.')

	parser.add_argument(
		'--output_path',
		default=DEFAULT_OUTPUT_PATH_STR,
		help='Path to save the results')

	parser.add_argument(
		'--data_path',
		default=DEFAULT_DATA_PATH_STR,
		help='Path to the integrated data file to process')

	parser.add_argument(
		'--dr_path',
		default=DEFAULT_DR_PATH_STR,
		help='Path to the file containing (domain / relation / range) info')

	return parser.parse_args()

if __name__ == '__main__':
	# set log and parse args
	set_logging()
	args = parse_argument()

	# # for displaying dataframe
	# pd.set_option('display.height', 1000)
	# pd.set_option('display.max_rows', 500)
	# pd.set_option('display.max_columns', 500)
	# pd.set_option('display.width', 1000)

	# read data and reformat it
	pd_data = DataProcessor(args.data_path).reformat_data()

	# save the reformatted data
	pd_data.to_csv(os.path.join(DEFAULT_OUTPUT_PATH_STR, DEFAULT_ALL_DATA_TXT_STR), sep='\t', index=False, header=None)

	# separate dataset into entities and relations
	ei = ExtractInfo(pd_data, args.dr_path)
	ei.save_all_entities(os.path.join(args.output_path, DEFAULT_ENTITIES_TXT_STR))
	ei.save_entity_full_names(os.path.join(args.output_path, DEFAULT_ENTITY_FULL_NAMES_TXT_STR))

	# split the dataset into specified folds
	sf = SplitFolds(pd_data, NUM_FOLDS, ei.get_entity_by_type('gene'))
	data_split_fold_dic = sf.split_into_folds()
	sf.save_folds(data_split_fold_dic, args.output_path)
