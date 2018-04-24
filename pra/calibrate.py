import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv
from sklearn.linear_model import LogisticRegression
relation = sys.argv[1]
scores_file = 'scores/'+relation
labels_file = 'queriesR_labels/'+relation
scores_file = 'scores/'+relation




if len(sys.argv)>2:
    scores_file = sys.argv[2]+'_scores/'+relation
    labels_file = 'queriesR_'+sys.argv[2]+'_labels/'+relation

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



