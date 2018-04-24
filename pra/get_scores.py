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
print(relation)
print(sys.argv[2])
use_calibration = False
if len(sys.argv)>2:
    calibration = sys.argv[2]
    if calibration=='use_calibration':
        use_calibration=True
        with open('calibrations/'+relation+'.pkl', 'rb') as pickle_file:
            log_reg = pickle.load(pickle_file)
queries_file = 'queriesR_test/'+relation
print(relation)
fname='predictions/'+relation
scores_file = 'scores/'+relation
if len(sys.argv)>2 and sys.argv[2] != 'use_calibration':
    queries_file = 'queriesR_'+sys.argv[2]+'/'+relation
    fname=sys.argv[2]+'_predictions/'+relation
    scores_file = sys.argv[2]+'_scores/'+relation

with open(fname) as f:
    lines = f.readlines()

with open(queries_file, "r") as l_file:
    queries = l_file.readlines()

scores = []
valid = []
for line in lines:
    tail_entity = queries.pop(0).strip().split('\t')[1]
    line.strip()
    words = line.split('\t')
    del words[0]
    del words[0]
    found = False
    for word in words:
        score_and_entity= word.split(',')
        tail_entity_star = '*'+tail_entity
        if score_and_entity[1].strip()==tail_entity.strip() or score_and_entity[1].strip()==tail_entity_star.strip()  :
            found = True
            break
    if found:
        # if use_calibration:
        #     print(np.array(float(score_and_entity[0])))
        #     score = log_reg.predict_proba( np.array(float(score_and_entity[0])).reshape(1, -1))
        # else:
        score = float(score_and_entity[0])
        # _file.write(str(score)+'\t1\n')
        scores.append(score)
        valid.append(1)
    else:
        # _file.write(str(0.0)+'\t0\n')
        scores.append(0.0)
        valid.append(0)


if use_calibration:
    scores_array = np.array(scores)
    valid_array =  np.array(valid)
    indices, = np.where(valid_array[:] > 0.)
    the_scores = scores_array[indices].reshape(-1, 1)
    scores =log_reg.predict_proba( the_scores)[:,1]
    scores_array[indices] = scores
else:
    scores_array = np.array(scores)
    valid_array =  np.array(valid)

with open(scores_file, "w") as _file:
    for i in range(np.shape(scores_array)[0]):
        _file.write(str(scores_array[i])+'\t'+str(valid_array[i])+'\n')

#



