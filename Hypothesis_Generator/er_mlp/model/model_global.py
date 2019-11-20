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

DIRECTORY = os.path.dirname(__file__)
ABS_PATH_ER_MLP = os.path.join(DIRECTORY, '../er_mlp_imp')
sys.path.insert(0, ABS_PATH_ER_MLP)
ABS_PATH_DATA = os.path.join(DIRECTORY, '../data_handler')
sys.path.insert(0, ABS_PATH_DATA)
ABS_PATH_UTILS = os.path.join(DIRECTORY, '../../utils')
sys.path.insert(0, ABS_PATH_UTILS)
