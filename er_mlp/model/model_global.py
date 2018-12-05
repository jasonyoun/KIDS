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

def set_logging():
	"""
	Configure logging.
	"""
	log.basicConfig(format='(%(levelname)s) %(filename)s: %(message)s', level=log.DEBUG)

	# set logging level to WARNING for matplotlib
	logger = log.getLogger('matplotlib')
	logger.setLevel(log.WARNING)

	# disable TF debugging logs
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
