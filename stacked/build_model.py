"""
Filename: build_model.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu

Description:
	Build stacked ensemble using AdaBoost.

To-do:
"""
import os
import sys
import argparse
import numpy as np
import pickle as pickle
import pandas as pd
directory = os.path.dirname(__file__)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  RandomizedSearchCV
abs_path_metrics= os.path.join(directory, '../utils')
sys.path.insert(0, abs_path_metrics)
from metrics import roc_auc_stats, pr_stats
from sklearn.model_selection import PredefinedSplit
import features
from imblearn.over_sampling import SMOTE
import configparser

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='build stacked ensemble')

	parser.add_argument(
		'--pra',
		metavar='pra_model (pra_model_2)',
		nargs='?',
		action='store',
		required=True,
		help='The pra models to add')

	parser.add_argument(
		'--er_mlp',
		metavar='er_mlp_model (er_mlp_model_2)',
		nargs='?',
		action='store',
		required=True,
		help='The er-mlp models to add')

	parser.add_argument(
		'--dir',
		metavar='dir',
		nargs='?',
		action='store',
		required=True,
		help='directory to store the model')

	return parser.parse_args()

if __name__ == "__main__":
	args = parse_argument()

	# paths
	model_instance_dir = 'model_instance'
	model_save_dir = os.path.join(model_instance_dir, args.dir)
	configuration = os.path.join(model_save_dir, '{}.ini'.format(args.dir))

	# read config
	config = configparser.ConfigParser()
	config.read('./' + configuration)

	RUN_RANDOM_SEARCH = config.getboolean('DEFAULT','RUN_RANDOM_SEARCH')
	TRAIN_DIR = config['DEFAULT']['TRAIN_DIR']
	TEST_DIR = config['DEFAULT']['TEST_DIR']
	DEV_DIR = config['DEFAULT']['DEV_DIR']
	USE_SMOTE_SAMPLING = config.getboolean('DEFAULT', 'USE_SMOTE_SAMPLING')
	LOG_REG_CALIBRATE = config.getboolean('DEFAULT', 'LOG_REG_CALIBRATE')
	F1_FOR_THRESHOLD = config.getboolean('DEFAULT', 'F1_FOR_THRESHOLD')
	RANDOM_SEARCH_COUNT = config.getint('DEFAULT', 'RANDOM_SEARCH_COUNT')
	RANDOM_SEARCH_PROCESSES = config.getint('DEFAULT', 'RANDOM_SEARCH_PROCESSES')

	ER_MLP_MODEL_DIR = config['DEFAULT']['ER_MLP_MODEL_DIR']
	PRA_MODEL_DIR = config['DEFAULT']['PRA_MODEL_DIR']

	RS_E_START = config.getint('RANDOM_SEARCH_ESTIMATORS', 'START')
	RS_E_END = config.getint('RANDOM_SEARCH_ESTIMATORS', 'END')
	RS_E_INCREMENT = config.getint('RANDOM_SEARCH_ESTIMATORS', 'INCREMENT')

	RS_LR_START = config.getfloat('RANDOM_SEARCH_LEARNING_RATE', 'START')
	RS_LR_END = config.getfloat('RANDOM_SEARCH_LEARNING_RATE', 'END')
	RS_LR_INCREMENT = config.getfloat('RANDOM_SEARCH_LEARNING_RATE', 'INCREMENT')

	def perform_randomized_search(pred_dic, train_x, train_y, test_x, test_y):

		def get_results_of_search(results, count=5):
			# roc_auc
			print("")
			print("")
			print("roc_auc")
			print("")
			for i in range(1, count + 1):
				runs = np.flatnonzero(results['rank_test_roc_auc'] == i)
				for run in runs:
					print("evaluation rank: {}".format(i))
					print("score: {}".format(results['mean_test_roc_auc'][run]))
					print("std: {}".format(results['std_test_roc_auc'][run]))
					print(results['params'][run])
					print("")

			# f1
			print("")
			print("")
			print("f1")
			print("")
			for i in range(1, count + 1):
				runs = np.flatnonzero(results['rank_test_f1'] == i)
				for run in runs:
					print("evaluation rank: {}".format(i))
					print("score: {}".format(results['mean_test_f1'][run]))
					print("std: {}".format(results['std_test_f1'][run]))
					print(results['params'][run])
					print("")

			# average precision
			print("")
			print("")
			print("average precision")
			print("")
			for i in range(1, count + 1):
				runs = np.flatnonzero(results['rank_test_average_precision'] == i)
				for run in runs:
					print("evaluation rank: {}".format(i))
					print("score: {}".format(results['mean_test_average_precision'][run]))
					print("std: {}".format(results['std_test_average_precision'][run]))
					print(results['params'][run])
					print("")

					# use average precision for reporting the results
					if i==1:
						AP_params = results['params'][run]

			# accuracy
			print("")
			print("")
			print("accuracy")
			print("")
			for i in range(1, count + 1):
				runs = np.flatnonzero(results['rank_test_accuracy'] == i)
				for run in runs:
					print("evaluation rank: {}".format(i))
					print("score: {}".format(results['mean_test_accuracy'][run]))
					print("std: {}".format(results['std_test_accuracy'][run]))
					print(results['params'][run])
					print("")

			return AP_params

		all_x = np.vstack((train_x, test_x))
		all_y = np.vstack((train_y, test_y))
		all_y = all_y.astype(int)

		# get train / test split indices for predefined split cross-validator
		train_indices = np.full(np.shape(train_x)[0], -1)
		test_indices = np.full(np.shape(test_x)[0], 0)
		test_fold = np.hstack((train_indices, test_indices))

		clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=0)
		param_distribution = {
			'learning_rate': np.arange(float(RS_LR_START), float(RS_LR_END), float(RS_LR_INCREMENT)),
			'n_estimators': np.arange(int(RS_E_START), int(RS_E_END), int(RS_E_INCREMENT))}

		random_search = RandomizedSearchCV(
			clf,
			param_distributions=param_distribution,
			n_iter=RANDOM_SEARCH_COUNT,
			n_jobs=RANDOM_SEARCH_PROCESSES,
			scoring=['roc_auc', 'f1', 'average_precision', 'accuracy'],
			cv=PredefinedSplit(test_fold),
			refit='average_precision')

		random_search.fit(all_x, all_y.ravel())

		return get_results_of_search(random_search.cv_results_)

	# read the previous prediction results
	pred_dic, train_x, train_y, predicates_train = features.get_x_y(TRAIN_DIR, ER_MLP_MODEL_DIR, PRA_MODEL_DIR)
	pred_dic, test_x, test_y, predicates_test = features.get_x_y(DEV_DIR, ER_MLP_MODEL_DIR, PRA_MODEL_DIR)

	# prediction results of adaboosst
	predictions_test = np.zeros_like(predicates_test, dtype=float)
	predictions_test = predictions_test.reshape((np.shape(predictions_test)[0], 1))

	model_dic = {}

	for k, i in pred_dic.items():
		test_indices, = np.where(predicates_test == i)

		# if we have matching indices
		if np.shape(test_indices)[0] != 0:
			test_x_predicate = test_x[test_indices]
			test_y_predicate = test_y[test_indices]

			train_indices, = np.where(predicates_train == i)
			if np.shape(train_indices)[0]!=0:
				train_x_predicate = train_x[train_indices]
				train_y_predicate = train_y[train_indices]
			else:
				print('No training data for predicate: {}'.format(k))
				sys.exit()

			if USE_SMOTE_SAMPLING:
				ros = SMOTE(ratio='minority')
				train_x_predicate, train_y_predicate = ros.fit_sample(train_x_predicate, train_y_predicate.ravel() )
				train_y_predicate = train_y_predicate.reshape(-1, 1)

			if RUN_RANDOM_SEARCH:
				AP_params = perform_randomized_search(pred_dic, train_x_predicate, train_y_predicate, test_x_predicate, test_y_predicate)
				config['RANDOM_SEARCH_BEST_PARAMS_{}'.format(k)] = {
					'n_estimators': AP_params['n_estimators'],
					'learning_rate': AP_params['learning_rate']}

			# build & fit model
			clf = AdaBoostClassifier(
				DecisionTreeClassifier(max_depth=1),
				n_estimators=config.getint('RANDOM_SEARCH_BEST_PARAMS_{}'.format(k), 'n_estimators'),
				learning_rate=config.getfloat('RANDOM_SEARCH_BEST_PARAMS_{}'.format(k), 'learning_rate'),
				random_state=0)

			clf.fit(train_x_predicate, train_y_predicate.ravel())

			# perform prediction
			preds = clf.predict_proba(test_x_predicate)[:, 1]
			preds = preds.reshape((np.shape(preds)[0], 1))
			predictions_test[test_indices] = preds[:]

			model_dic[i] = clf

	score = pr_stats(len(pred_dic), test_y, predictions_test, predicates_test, pred_dic)
	roc_ = roc_auc_stats(len(pred_dic), test_y, predictions_test, predicates_test, pred_dic)
	print('score: {}'.format(score))
	print('roc_: {}'.format(roc_))

	with open(os.path.join(model_save_dir, 'model.pkl'), 'wb') as output:
		pickle.dump(model_dic, output, pickle.HIGHEST_PROTOCOL)

	with open('./' + configuration, 'w') as configfile:
		config.write(configfile)
