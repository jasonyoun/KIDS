"""
Filename: analyze_inconsistency_validation.py

Authors:
	Jason Youn -jyoun@ucdavis.edu

Description:

To-do:
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging as log
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix, f1_score

# default paths
DEFAULT_DATA_DIR_STR = './data/inconsistency_validation'
DEFAULT_OUTPUT_DIR_STR = './output'

# default file names
DEFAULT_THREE_SRC_INCONSISTENCY_VALIDATION_XLSX_STR = 'three_sources_inconsistency_validation.xlsx'
DEFAULT_ALL_SRC_INCONSISTENCY_VALIDATION_XLSX_STR = 'all_sources_inconsistency_validation.xlsx'

CRA_STR = 'confers resistance to antibiotic'
CNRA_STR = 'confers no resistance to antibiotic'

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
	parser = argparse.ArgumentParser(description='Analyze inconsistency validation experimental results.')

	parser.add_argument(
		'--output_path',
		default=DEFAULT_OUTPUT_DIR_STR,
		help='Directory to save the results')

	parser.add_argument(
		'--data_dir_path',
		default=DEFAULT_DATA_DIR_STR,
		help='Directory containing the validation files to analyze')

	parser.add_argument(
		'--validation_file',
		default=DEFAULT_ALL_SRC_INCONSISTENCY_VALIDATION_XLSX_STR,
		help='Filename of the .xlsx file containing validation info')

	return parser.parse_args()


def calculate_recall(tp, fn):
	if tp + fn == 0:
		return 0
	else:
		return tp / (tp + fn)

def calculate_precision(tp, fp):
	if tp + fp == 0:
		return 0
	else:
		return tp / (tp + fp)

def calculate_f1(recall, precision):
	if precision + recall == 0:
		return 0
	else:
		return (2 * precision * recall) / (precision + recall)

if __name__ == '__main__':
	# set log and parse args
	set_logging()
	args = parse_argument()


	file_path = os.path.join(args.data_dir_path, args.validation_file)
	pd_data = pd.read_excel(file_path)

	pd_data['Resolution label']  = 0
	idx = pd_data['Predicate'].str.match(CRA_STR)
	pd_data.loc[idx, 'Resolution label'] = 1

	pd_data['Validation label']  = 0
	idx = pd_data['Validation'].str.match(CRA_STR)
	pd_data.loc[idx, 'Validation label'] = 1


	belief_score = pd_data['Belief']
	belief_difference = pd_data['Belief difference']
	bd_unique = np.sort(belief_difference.unique())

	bd_list = []
	f1_list = []
	for bd in bd_unique:
		new_pd_data = pd_data[pd_data['Belief difference'] >= bd]
		resolution_label = new_pd_data['Resolution label']
		validation_label = new_pd_data['Validation label']

		cm_result = confusion_matrix(validation_label, resolution_label)

		if cm_result.shape == (1, 1):
			log.warning('Confusion matrix output has size {} for belief difference {}.'.format(cm_result.shape, bd))
			continue

		tp = cm_result[1, 1]
		fp = cm_result[0, 1]
		fn = cm_result[1, 0]
		tn = cm_result[0, 0]

		recall = calculate_recall(tp, fn)
		precision = calculate_precision(tp, fp)
		f1 = calculate_f1(recall, precision)

		bd_list.append(bd)
		f1_list.append(f1)

	print(bd_list)
	print(f1_list)

	plt.figure()
	plt.plot(bd_list, f1_list)







	bs_unique = np.sort(belief_score.unique())

	bs_list = []
	f1_list = []
	for bs in bs_unique:
		new_pd_data = pd_data[pd_data['Belief'] >= bs]
		resolution_label = new_pd_data['Resolution label']
		validation_label = new_pd_data['Validation label']

		cm_result = confusion_matrix(validation_label, resolution_label)

		if cm_result.shape == (1, 1):
			log.warning('Confusion matrix output has size {} for belief score {}.'.format(cm_result.shape, bs))
			continue

		tp = cm_result[1, 1]
		fp = cm_result[0, 1]
		fn = cm_result[1, 0]
		tn = cm_result[0, 0]

		recall = calculate_recall(tp, fn)
		precision = calculate_precision(tp, fp)
		f1 = calculate_f1(recall, precision)

		bs_list.append(bs)
		f1_list.append(f1)

	print(bs_list)
	print(f1_list)

	plt.figure()
	plt.plot(bs_list, f1_list)
	plt.show()

