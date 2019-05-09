"""
Filename: determine_thresholds.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
	Determine the threshold.

To-do:
	1. why set label to 0 now instead of -1?
"""
import model_global
import os
import argparse
import configparser
import numpy as np
import pickle as pickle
import tensorflow as tf
import logging as log
from er_mlp import ERMLP
from data_processor import DataProcessor

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Determine the thresholds.')

	parser.add_argument(
		'--dir',
		metavar='dir',
		nargs='?',
		default='./',
		help='Base directory')
	parser.add_argument(
		'--logfile',
		default='',
		help='Path to save the log')

	return parser.parse_args()

if __name__ == '__main__':
	# set log and parse args
	args = parse_argument()
	model_global.set_logging(args.logfile)

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
	F1_FOR_THRESHOLD = config.getboolean('DEFAULT', 'F1_FOR_THRESHOLD')

	with tf.Session() as sess:
		# load the saved parameters
		with open(os.path.join(MODEL_SAVE_DIRECTORY, 'params.pkl'), 'rb') as f:
			params = pickle.load(f)

		# some parameters
		entity_dic = params['entity_dic']
		pred_dic = params['pred_dic']

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
			'act_function':ACT_FUNCTION,
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
		dev_df = processor.load(os.path.join(DATA_PATH, 'dev.txt'))
		indexed_data_dev = processor.create_indexed_triplets_test(dev_df.as_matrix(), entity_dic, pred_dic)
		indexed_data_dev[:, 3][indexed_data_dev[:, 3] == -1] = 0

		# find the threshold
		thresholds = er_mlp.determine_threshold(sess, indexed_data_dev, f1=F1_FOR_THRESHOLD)

		log.debug('thresholds: {}'.format(thresholds))

		# define what to save
		save_object = {
			'entity_dic': entity_dic,
			'pred_dic': pred_dic,
			'thresholds': thresholds
		}

		if hasattr(params, 'thresholds_calibrated'):
			save_object['thresholds_calibrated'] = params['thresholds_calibrated']

		if hasattr(params, 'calibrated_models'):
			save_object['calibrated_models'] = params['calibrated_models']

		if WORD_EMBEDDING:
			save_object['indexed_entities'] = indexed_entities
			save_object['indexed_predicates'] = indexed_predicates
			save_object['num_pred_words'] = num_pred_words
			save_object['num_entity_words'] = num_entity_words

		# save the parameters
		with open(os.path.join(MODEL_SAVE_DIRECTORY, 'params.pkl'), 'wb') as output:
			pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)
