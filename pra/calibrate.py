import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv
from sklearn.linear_model import LogisticRegression
import argparse

parser = argparse.ArgumentParser(description='calibrate')
parser.add_argument('--predicate', nargs='?',required=True,
                    help='the predicate that we will get the scores for')
parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')

args = parser.parse_args()
print(args.dir)

relation = args.predicate

queries_file = args.dir+'/queriesR_test/'+relation
print(relation)
fname=args.dir+'/predictions/'+relation
scores_file = args.dir+'/scores/'+relation

labels_file = args.dir+'/queriesR_labels/'+relation


with open(labels_file, "r") as l_file:
    labels = l_file.readlines()

with open(scores_file, "r") as _file:
    scores = _file.readlines()

scores_split = []
for i in scores:
    row = i.split('\t')
    row = [float(row[0].strip()),int(float(row[1].strip()))]
    scores_split.append(row)

labels_split = []
for i in labels:
    labels_split.append(int(i.split('\t')[2].strip()))



scores_array = np.array(scores_split)
labels_array = np.array(labels_split)
indices, = np.where(scores_array[:,1] != 0)
scores_array = scores_array[indices]
labels_array = labels_array[indices]
log_reg = LogisticRegression()
log_reg.fit( scores_array[:,0].reshape(-1, 1), labels_array.ravel() )  


with open('calibrations/'+relation+'.pkl', 'wb') as output:
    pickle.dump(log_reg, output, pickle.HIGHEST_PROTOCOL)



