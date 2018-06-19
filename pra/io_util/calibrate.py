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
from sklearn.isotonic import IsotonicRegression as IR
from imblearn.over_sampling import RandomOverSampler,SMOTE

parser = argparse.ArgumentParser(description='calibrate')
parser.add_argument('--predicate', nargs='?',required=True,
                    help='the predicate that we will get the scores for')
parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')
parser.add_argument('--freebase',action='store_const',default=False,const=True)

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
print('scores array before')
print(np.shape(scores_array))
labels_array = np.array(labels_split)
indices, = np.where(scores_array[:,1] != 0)
scores_array = scores_array[indices]
print('scores array after')
print(np.shape(scores_array))
labels_array = labels_array[indices]
labels_array[:][labels_array[:] == -1] = 0
# print(labels_array)
print(np.shape(scores_array))
print(np.shape(labels_array))
ir = IR(out_of_bounds='clip'  )
if args.freebase:
	X_train = scores_array[:,0]
	y_train = labels_array
else:
    # X_train = scores_array[:,0]
    # y_train = labels_array
	ros = SMOTE(ratio='minority')
	X_train, y_train = ros.fit_sample(scores_array[:,0].reshape(-1, 1), labels_array.ravel() )

print(np.shape(X_train))
print(np.shape(y_train))
ir.fit( X_train.ravel(), y_train.ravel()  )

# log_reg = LogisticRegression()
# log_reg.fit( X_train.reshape(-1, 1), y_train.ravel() )  


with open('calibrations/'+relation+'.pkl', 'wb') as output:
    pickle.dump(ir, output, pickle.HIGHEST_PROTOCOL)



