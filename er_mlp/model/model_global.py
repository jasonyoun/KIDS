"""
Filename: model_global.py

Authors:
	Jason Youn - jyoun@ucdavis.edu

Description:
	Gobal file to be imported by all files under the model directory.

To-do:
"""
import os
import sys
import logging as log

directory = os.path.dirname(__file__)
abs_path_er_mlp = os.path.join(directory, '../er_mlp_imp')
sys.path.insert(0, abs_path_er_mlp)
abs_path_data= os.path.join(directory, '../data_handler')
sys.path.insert(0, abs_path_data)
abs_path_metrics= os.path.join(directory, '../../utils')
sys.path.insert(0, abs_path_metrics)

def set_logging(logfile=''):
	"""
	Configure logging.
	"""
	# create logger
	logger = log.getLogger()
	logger.setLevel(log.DEBUG)

	# create formatter
	formatter = log.Formatter('%(asctime)s %(levelname)s %(filename)s: %(message)s')

	# create and set file handler if requested
	if len(logfile) > 0:
		fh = log.FileHandler(logfile)
		fh.setLevel(log.DEBUG)
		fh.setFormatter(formatter)
		logger.addHandler(fh)

	# create and set console handler
	ch = log.StreamHandler()
	ch.setLevel(log.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	# set logging level to WARNING for matplotlib
	matplotlib_logger = log.getLogger('matplotlib')
	matplotlib_logger.setLevel(log.WARNING)

	# disable TF debugging logs
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
