import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score

relation = sys.argv[1]

scores_file = 'scores/'+relation

thresholds_file = 'thresholds/'+relation

classifications_file = 'classifications/'+relation


with open(scores_file, "r") as _file:
	scores = _file.readlines()

with open(thresholds_file, "r") as _file:
    threshold = float(_file.readline().strip())



scores = [x.strip().split('\t') for x in scores] 
classifications = []


with open(classifications_file, "w") as _file:
    for score in scores:
        if float(score[0]) > threshold:
            _file.write('1\t'+str(score[0])+'\n')
        else:
            _file.write('0\t'+str(score[0])+'\n')



