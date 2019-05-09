"""
Filename: predict.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
	

To-do:
	1. check each line and understand
"""
import model_global
import os
import argparse
import configparser
import numpy as np
import pickle as pickle
import tensorflow as tf
import logging as log
from data_processor import DataProcessor
from er_mlp import ERMLP

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Evaluate network.')

	parser.add_argument(
		'--dir',
		metavar='dir',
		nargs='?',
		default='./',
		help='Base directory')
	parser.add_argument(
		'--use_calibration',
		action='store_const',
		default=False,
		const=True)
	parser.add_argument(
		'--predict_file',
		required=True)
	parser.add_argument(
		'--logfile',
		default='',
		help='Path to save the log')

	return parser.parse_args()

def calibrate_probabilties(predictions_list_test, num_preds, calibration_models, predicates_test, pred_dic):
	for k, i in pred_dic.items():
		indices, = np.where(predicates_test == i)

		if np.shape(indices)[0] != 0:
			predictions_predicate = predictions_list_test[indices]
			clf = calibration_models[i]

			if LOG_REG_CALIBRATE:
				p_calibrated = clf.predict_proba(predictions_predicate.reshape(-1, 1))[:, 1]
			else:
				p_calibrated = clf.transform(predictions_predicate.ravel())

			predictions_list_test[indices] = np.reshape(p_calibrated, (-1, 1))

	return predictions_list_test

if __name__ == '__main__':
	# set log and parse args
	args = parse_argument()
	model_global.set_logging(args.logfile)

	# some init
	calibrated = args.use_calibration

	# directory and filename setup
	model_instance_dir = 'model_instance'
	model_save_dir = os.path.join(model_instance_dir, args.dir)
	configuration = os.path.join(model_save_dir, '{}.ini'.format(args.dir))

	# read configuration
	config = configparser.ConfigParser()
	config.read(configuration)

	WORD_EMBEDDING = config.getboolean('DEFAULT', 'WORD_EMBEDDING')
	MODEL_SAVE_DIRECTORY = model_save_dir
	DATA_PATH = config['DEFAULT']['DATA_PATH']
	WORD_EMBEDDING = config.getboolean('DEFAULT', 'WORD_EMBEDDING')
	TRAINING_EPOCHS = config.getint('DEFAULT', 'TRAINING_EPOCHS')
	BATCH_SIZE = config.getint('DEFAULT', 'BATCH_SIZE')
	DISPLAY_STEP =  config.getint('DEFAULT', 'DISPLAY_STEP')
	EMBEDDING_SIZE = config.getint('DEFAULT', 'EMBEDDING_SIZE')
	LAYER_SIZE = config.getint('DEFAULT', 'LAYER_SIZE')
	LEARNING_RATE = config.getfloat('DEFAULT', 'LEARNING_RATE')
	CORRUPT_SIZE = config.getint('DEFAULT', 'CORRUPT_SIZE')
	LAMBDA = config.getfloat('DEFAULT', 'LAMBDA')
	OPTIMIZER = config.getint('DEFAULT', 'OPTIMIZER')
	ACT_FUNCTION = config.getint('DEFAULT', 'ACT_FUNCTION')
	ADD_LAYERS = config.getint('DEFAULT', 'ADD_LAYERS')
	DROP_OUT_PERCENT = config.getfloat('DEFAULT', 'ADD_LAYERS')
	PREDICT_FILE = args.predict_file
	LOG_REG_CALIBRATE = config.getboolean('DEFAULT', 'LOG_REG_CALIBRATE')

	with tf.Session() as sess:
		# load the saved parameters
		with open(os.path.join(MODEL_SAVE_DIRECTORY, 'params.pkl'), 'rb') as f:
			params = pickle.load(f)

		# some parameters
		entity_dic = params['entity_dic']
		pred_dic = params['pred_dic']
		thresholds = params['thresholds']

		if calibrated:
			calibration_models = params['calibrated_models']
			thresholds = params['thresholds_calibrated']

		num_entities = len(entity_dic)
		num_preds = len(pred_dic)

		er_mlp_params = {
			'word_embedding': WORD_EMBEDDING,
			'embedding_size': EMBEDDING_SIZE,
			'layer_size': LAYER_SIZE,
			'corrupt_size': CORRUPT_SIZE,
			'lambda': LAMBDA,
			'num_entities': num_entities,
			'num_preds': num_preds,
			'learning_rate': LEARNING_RATE,
			'batch_size': BATCH_SIZE,
			'add_layers': ADD_LAYERS,
			'act_function': ACT_FUNCTION,
			'drop_out_percent': DROP_OUT_PERCENT
		}

		if WORD_EMBEDDING:
			num_entity_words = params['num_entity_words']
			num_pred_words = params['num_pred_words']
			indexed_entities = params['indexed_entities']
			indexed_predicates = params['indexed_predicates']
			er_mlp_params['num_entity_words'] = num_entity_words
			er_mlp_params['num_pred_words'] = num_pred_words
			er_mlp_params['indexed_entities'] = indexed_entities
			er_mlp_params['indexed_predicates'] = indexed_predicates

		# init ERMLP class using the parameters defined above
		er_mlp = ERMLP(
			er_mlp_params,
			sess,
			meta_graph=os.path.join(MODEL_SAVE_DIRECTORY, 'model.meta'),
			model_restore=os.path.join(MODEL_SAVE_DIRECTORY, 'model'))

		processor = DataProcessor()
		test_df = processor.load(os.path.join(DATA_PATH, PREDICT_FILE))

		if (test_df.shape[1] == 4):
			indexed_data_test = processor.create_indexed_triplets_test(test_df.as_matrix(), entity_dic, pred_dic)
		else:
			indexed_data_test = processor.create_indexed_triplets_training(test_df.as_matrix(), entity_dic, pred_dic)

		data_test = indexed_data_test[:, :3]
		predicates_test = indexed_data_test[:, 1]

		predictions_list_test = sess.run(er_mlp.test_predictions, feed_dict={er_mlp.test_triplets: data_test})

		if calibrated:
			predictions_list_test = calibrate_probabilties(predictions_list_test, num_preds, calibration_models, predicates_test, pred_dic)

		classifications_test = er_mlp.classify(predictions_list_test, thresholds, predicates_test)
		classifications_test = np.array(classifications_test).astype(int)
		classifications_test = classifications_test.reshape((np.shape(classifications_test)[0], 1))

		c = np.dstack((classifications_test, predictions_list_test))
		c = np.squeeze(c)

		if (test_df.shape[1] == 4):
			labels_test = indexed_data_test[:, 3]
			labels_test = labels_test.reshape((np.shape(labels_test)[0], 1))

			c = np.concatenate((c, labels_test), axis=1)

		predict_folder = os.path.splitext(PREDICT_FILE)[0]
		predict_folder = os.path.join(MODEL_SAVE_DIRECTORY, predict_folder)

		if not os.path.exists(predict_folder):
			os.makedirs(predict_folder)

		with open(os.path.join(predict_folder, 'predictions.txt'), 'w') as _file:
			for i in range(np.shape(c)[0]):
				if (test_df.shape[1] == 4):
					_file.write('predicate: ' + str(predicates_test[i]) + '\tclassification: ' + str(int(c[i][0])) + '\tprediction: ' + str(c[i][1]) + '\tlabel: ' + str(int(c[i][2])) + '\n')
				else:
					_file.write('predicate: ' + str(predicates_test[i]) + '\tclassification: ' + str(int(c[i][0])) + '\tprediction: ' + str(c[i][1]) + '\n')
