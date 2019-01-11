"""
Filename: evaluate.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu

Description:
	Evaluate the performance of the stacked model.

To-do:
"""
import os
import sys
import argparse
import features
import configparser
import numpy as np
import pickle as pickle
directory = os.path.dirname(__file__)
from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
abs_path_metrics= os.path.join(directory, '../utils')
sys.path.insert(0, abs_path_metrics)
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats, save_results

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='build stacked ensemble')

	parser.add_argument(
		'--dir',
		metavar='dir',
		nargs='?',
		action='store',
		required=True,
		help='directory to store the model')

	parser.add_argument(
		'--use_calibration',
		action='store_const',
		default=False,
		const=True)

	return parser.parse_args()

def classify(predictions_list, threshold, predicates):
	classifications = []

	for i in range(len(predictions_list)):
		if(predictions_list[i] >= threshold):
			classifications.append(1)
		else:
			classifications.append(0)

	return np.array(classifications)

if __name__ == "__main__":
	args = parse_argument()
	config = configparser.ConfigParser()

	model_instance_dir = 'model_instance'
	model_save_dir = os.path.join(model_instance_dir, args.dir)
	configuration = os.path.join(model_save_dir, '{}.ini'.format(args.dir))
	config.read('./' + configuration)

	er_mlp_model_dir = config['DEFAULT']['er_mlp_model_dir']
	pra_model_dir = config['DEFAULT']['pra_model_dir']
	LOG_REG_CALIBRATE = config.getboolean('DEFAULT','LOG_REG_CALIBRATE')
	TEST_DIR = config['DEFAULT']['TEST_DIR']

	use_calibration = args.use_calibration
	if use_calibration:
		with open(os.path.join(model_save_dir, 'calibrations.pkl'), 'rb') as pickle_file:
			clf_dic = pickle.load(pickle_file)

	if use_calibration:
		fn = open(os.path.join(model_save_dir, 'calibrated_threshold.pkl'), 'rb')
	else:
		fn = open(os.path.join(model_save_dir, 'threshold.pkl'), 'rb')
	threshold = pickle.load(fn)

	fn = open(os.path.join(model_save_dir, 'model.pkl'),'rb')
	model_dic = pickle.load(fn)
	pred_dic, test_x, test_y, predicates_test = features.get_x_y(TEST_DIR, er_mlp_model_dir, pra_model_dir)

	# test_y[:][test_y[:] == -1] = 0
	best_thresholds = np.zeros(len(pred_dic))
	probabilities = np.zeros_like(predicates_test, dtype=float)
	probabilities = probabilities.reshape((np.shape(probabilities)[0], 1))
	standard_classifications = np.zeros_like(predicates_test)
	threshold_classifications = np.zeros_like(predicates_test)

	for k, i in pred_dic.items():
		indices, = np.where(predicates_test == i)
		if np.shape(indices)[0] != 0:
			clf = model_dic[i]
			X = test_x[indices]
			y = test_y[indices]
			predicates = predicates_test[indices]

			preds = clf.predict_proba(X)[:, 1]
			preds_class = clf.predict(X)
			preds = preds.reshape((np.shape(preds)[0], 1))
			probabilities[indices] = preds
			standard_classifications[indices] = preds_class
			threshold_classifications[indices] = classify(preds, threshold[i], predicates)

	if use_calibration:
		for k, i in pred_dic.items():
			indices, = np.where(predicates_test == i)
			if np.shape(indices)[0] != 0:
				clf = clf_dic[i]
				X = probabilities[indices]

				if LOG_REG_CALIBRATE:
					preds = clf.predict_proba(X)[:, 1]
				else:
					preds = clf.transform(X.ravel())

				preds = preds.reshape((np.shape(preds)[0], 1))
				probabilities[indices] = preds

		# scores =log_reg.predict_proba( probabilities.reshape(-1,1))[:,1]
		# probabilities = scores

	classifications = standard_classifications
	# classifications = threshold_classifications

	results = {}
	results['predicate'] = {}
	plot_roc(len(pred_dic), test_y, probabilities, predicates, pred_dic, model_save_dir)
	plot_pr(len(pred_dic), test_y, probabilities, predicates, pred_dic, model_save_dir)
	auc_test = roc_auc_stats(len(pred_dic), test_y, probabilities, predicates, pred_dic)
	ap_test = pr_stats(len(pred_dic), test_y, probabilities, predicates, pred_dic)

	fl_measure_test = f1_score(test_y, classifications)
	accuracy_test = accuracy_score(test_y, classifications)
	recall_test = recall_score(test_y, classifications)
	precision_test = precision_score(test_y, classifications)
	confusion_test = confusion_matrix(test_y, classifications)

	results['overall'] = {
		'map': ap_test,
		'roc_auc': auc_test,
		'f1': fl_measure_test,
		'accuracy': accuracy_test,
		'cm': confusion_test,
		'precision': precision_test,
		'recall': recall_test
	}

	for i in range(len(pred_dic)):
		for key, value in pred_dic.items():
			if value == i:
				pred_name = key

		indices, = np.where(predicates == i)

		if np.shape(indices)[0] != 0:
			print(indices)
			print(classifications)
			classifications_predicate = classifications[indices]
			predicate_predictions = probabilities[indices]
			labels_predicate = test_y[indices]
			fl_measure_predicate = f1_score(labels_predicate, classifications_predicate)
			accuracy_predicate = accuracy_score(labels_predicate, classifications_predicate)
			recall_predicate = recall_score(labels_predicate, classifications_predicate)
			precision_predicate = precision_score(labels_predicate, classifications_predicate)
			confusion_predicate = confusion_matrix(labels_predicate, classifications_predicate)

			print(' - predicate f1 measure for {}: {}'.format(pred_name, fl_measure_predicate))
			print(' - predicate accuracy for {}: {}'.format(pred_name, accuracy_predicate))
			print(' - predicate recall for {}: {}'.format(pred_name, recall_predicate))
			print(' - predicate precision for {}: {}'.format(pred_name, precision_predicate))
			print(' - predicate confusion matrix for {}:'.format(pred_name))
			print(confusion_predicate)
			print(' ')

			fpr_pred, tpr_pred , _ = roc_curve(labels_predicate.ravel(), predicate_predictions.ravel())
			roc_auc_pred = auc(fpr_pred, tpr_pred)
			ap_pred = average_precision_score(labels_predicate.ravel(), predicate_predictions.ravel())

			results['predicate'][pred_name] = {
				'map': ap_pred,
				'roc_auc': roc_auc_pred,
				'f1': fl_measure_predicate,
				'accuracy': accuracy_predicate,
				'cm': confusion_predicate,
				'precision': precision_predicate,
				'recall': recall_predicate
			}

	directory = os.path.join(model_save_dir, 'test')
	if not os.path.exists(directory):
		os.makedirs(directory)

	_file = os.path.join(directory, 'classifications_stacked.txt')
	with open(_file, 'w') as t_f:
		for row in classifications:
			t_f.write(str(row) + '\n')

	save_results(results, directory)

	print('test f1 measure: {}'.format(fl_measure_test))
	print('test accuracy: {}'.format(accuracy_test))
	print('test precision: {}'.format(precision_test))
	print('test recall: {}'.format(recall_test))
	print('test confusion matrix:')
	print('test auc: {}'.format(auc_test))
	print('test ap: {}'.format(ap_test))
	print(confusion_test)
	print(ap_test)
