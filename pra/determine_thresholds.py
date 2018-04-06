import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score





def compute_threshold( predictions_list, dev_labels, f1=False):
    min_score = np.min(predictions_list) 
    max_score = np.max(predictions_list) 
    best_threshold = min_score
    best_accuracy = -1
    score = min_score
    increment = 0.01
    while(score <= max_score):
        predictions = (predictions_list >= score) * 2 -1
        accuracy = np.mean(predictions == dev_labels)
        if f1:
            accuracy = f1_score(dev_labels,predictions)
        # accuracy = np.mean(predictions == dev_labels)
        if accuracy > best_accuracy:
            best_threshold = score
            best_accuracy = accuracy
        score += increment
    return best_threshold


relation = sys.argv[1]

scores_file = 'scores/'+relation

thresholds_file = 'thresholds/'+relation

labels_file = 'queriesR_labels/'+relation
if len(sys.argv)>2:
    labels_file = 'queriesR_'+sys.argv[2]+'_labels/'+relation

with open(labels_file, "r") as l_file:
    labels = l_file.readlines()

with open(scores_file, "r") as _file:
	scores = _file.readlines()


scores = [x.strip().split('\t') for x in scores] 
labels = [x.strip().split('\t') for x in labels] 
scores = np.array(scores)
labels = np.array(labels)
labels = labels[:,2]
labels= labels.astype(np.int)
labels[:][labels[:] == 0] = -1
threshold = compute_threshold(scores[:,0].astype(np.float),labels,f1=True)
with open(thresholds_file, "w") as _file:
	 _file.write(str(threshold)+'\n')



