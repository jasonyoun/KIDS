import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='determine the threshold')
parser.add_argument('--predicate', nargs='?',required=True,
                    help='the predicate that we will get the scores for')
parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')
parser.add_argument('--use_calibration',action='store_const',default=False,const=True)

args = parser.parse_args()
print(args.dir)
use_calibration = args.use_calibration


def compute_threshold( predictions_list, dev_labels, f1=True):
    predictions_list = predictions_list.reshape(-1,1)
    dev_labels = dev_labels.reshape(-1,1)
    both = np.column_stack((predictions_list,dev_labels))
    both = both[both[:,0].argsort()]
    predictions_list = both[:,0].ravel()
    dev_labels = both[:,1].ravel()
    best_accuracy = -1
    accuracies = np.zeros(np.shape(predictions_list))
    for i in range(np.shape(predictions_list)[0]):
        score = predictions_list[i]
        predictions = (predictions_list >= score) * 2 -1
        accuracy = accuracy_score(predictions, dev_labels)
        if f1:
            accuracy = f1_score(dev_labels,predictions)
        accuracies[i] = accuracy
    indices=np.argmax(accuracies)
    best_threshold = np.mean(predictions_list[indices])
    return best_threshold



relation = args.predicate

scores_file = args.dir+'/scores/'+relation

thresholds_file = args.dir+'/thresholds/'+relation
if use_calibration:
    thresholds_file = args.dir+'/thresholds_calibration/'+relation
labels_file = args.dir+'/queriesR_labels/'+relation

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



