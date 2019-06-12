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

DIRECTORY = os.path.dirname(__file__)
ABS_PATH_ER_MLP = os.path.join(DIRECTORY, '../er_mlp_imp')
sys.path.insert(0, ABS_PATH_ER_MLP)
ABS_PATH_DATA = os.path.join(DIRECTORY, '../data_handler')
sys.path.insert(0, ABS_PATH_DATA)
ABS_PATH_METRICS = os.path.join(DIRECTORY, '../../utils')
sys.path.insert(0, ABS_PATH_METRICS)

def set_logging(logfile_str=''):
    """
    Configure logging.

    Inputs: path to save the log file
    """
    # create logger
    logger = log.getLogger()
    logger.setLevel(log.DEBUG)

    # create formatter
    formatter = log.Formatter('%(asctime)s %(levelname)s %(filename)s: %(message)s')

    # create and set file handler if requested
    if logfile_str:
        file_handler = log.FileHandler(logfile_str)
        file_handler.setLevel(log.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # create and set console handler
    stream_handler = log.StreamHandler()
    stream_handler.setLevel(log.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # set logging level to WARNING for matplotlib
    matplotlib_logger = log.getLogger('matplotlib')
    matplotlib_logger.setLevel(log.WARNING)

    # disable TF debugging logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
