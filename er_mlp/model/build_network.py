"""
Filename: build_network.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
	Build and run the ER MLP network.

To-do:
"""
import model_global
import os
import argparse
import configparser
import er_mlp_max_margin
import logging as log
from data_processor import DataProcessor

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Build, run, and test ER MLP.')

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
	log.info('Reading the configuration from \'{}\'...'.format(configuration))

	config = configparser.ConfigParser()
	config.read(configuration)

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
	DROP_OUT_PERCENT = config.getfloat('DEFAULT', 'DROP_OUT_PERCENT')
	DATA_PATH = config['DEFAULT']['DATA_PATH']
	SAVE_MODEL = config.getboolean('DEFAULT', 'SAVE_MODEL')
	MODEL_SAVE_DIRECTORY = model_save_dir
	TRAIN_FILE = config['DEFAULT']['TRAIN_FILE']
	SEPARATOR = config['DEFAULT']['SEPARATOR']
	F1_FOR_THRESHOLD = config.getboolean('DEFAULT', 'F1_FOR_THRESHOLD')
	USE_SMOLT_SAMPLING = config.getboolean('DEFAULT', 'USE_SMOLT_SAMPLING')
	LOG_REG_CALIBRATE = config.getboolean('DEFAULT', 'LOG_REG_CALIBRATE')
	MARGIN = config.getfloat('DEFAULT', 'MARGIN')

	params = {
		'WORD_EMBEDDING': WORD_EMBEDDING,
		'TRAINING_EPOCHS': TRAINING_EPOCHS,
		'BATCH_SIZE': BATCH_SIZE,
		'DISPLAY_STEP': DISPLAY_STEP,
		'EMBEDDING_SIZE': EMBEDDING_SIZE,
		'LAYER_SIZE': LAYER_SIZE,
		'LEARNING_RATE': LEARNING_RATE,
		'CORRUPT_SIZE': CORRUPT_SIZE,
		'LAMBDA': LAMBDA,
		'OPTIMIZER': OPTIMIZER,
		'ACT_FUNCTION': ACT_FUNCTION,
		'ADD_LAYERS': ADD_LAYERS,
		'DROP_OUT_PERCENT': DROP_OUT_PERCENT,
		'DATA_PATH': DATA_PATH,
		'SAVE_MODEL': SAVE_MODEL,
		'MODEL_SAVE_DIRECTORY': MODEL_SAVE_DIRECTORY,
		'TRAIN_FILE': TRAIN_FILE,
		'SEPARATOR': SEPARATOR,
		'F1_FOR_THRESHOLD': F1_FOR_THRESHOLD,
		'USE_SMOLT_SAMPLING': USE_SMOLT_SAMPLING,
		'LOG_REG_CALIBRATE': LOG_REG_CALIBRATE,
		'MARGIN': MARGIN
	}

	# run the model
	er_mlp_max_margin.run_model(params)
