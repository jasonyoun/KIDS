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


parser = argparse.ArgumentParser(description='parse and generate the scores file')
parser.add_argument('--use_calibration',action='store_const',default=False,const=True)
parser.add_argument('--predicate', nargs='?',required=True,
                    help='the predicate that we will get the scores for')
parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')

args = parser.parse_args()
print(args.dir)
relation = args.predicate
print(relation)
use_calibration = args.use_calibration
if use_calibration:
    with open('calibrations/'+relation+'.pkl', 'rb') as pickle_file:
        log_reg = pickle.load(pickle_file)

queries_file = args.dir+'/queriesR_test/'+relation
queries_tail = args.dir+'/queriesR_tail/'+relation
print(relation)
fname=args.dir+'/predictions/'+relation
scores_file = args.dir+'/scores/'+relation

with open(fname) as f:
    lines = f.readlines()

with open(queries_file, "r") as l_file:
    queries = l_file.readlines()

with open(queries_tail, "r") as l_file:
    queries_tail = l_file.readlines()

scores = []
valid = []
entities_scores_dic = {}
for line in lines:
    words = line.split('\t')
    subject = words[0]
    if subject not in entities_scores_dic:
        entities_scores_dic[subject] = {}
    del words[0]
    del words[0]
    # for score_and_entity in words
    # if len(words)>2:
        # print(words[2])
        # print(words[2].split(','))
    for score_and_entity in words:
        score_and_entity = score_and_entity.split(',')
        # print(score_and_entity)
        # print(score_and_entity[1])
        # print(score_and_entity)
        # print(score_and_entity[1])
        entity = score_and_entity[1].replace('*','').strip()
        score = float(score_and_entity[0].strip())
        entities_scores_dic[subject][entity]= score
    # if len(words)>3:
    #     print(words[3])
    #     for score_and_entity in words[3].split(' '):
    #         score_and_entity = score_and_entity.split(',')
    #         entity = score_and_entity[1].replace('*','').strip()
    #         score = float(score_and_entity[0].strip())
    #         print(subject)
    #         print(entity)
    #         entities_scores_dic[subject][entity]= score
# print(entities_scores_dic)

scores = []
valid = []
for line in queries_tail:
    subject = line.split('\t')[0].strip()
    tail = line.split('\t')[1].strip()
    if tail in entities_scores_dic[subject]:
        scores.append(entities_scores_dic[subject][tail])
        valid.append(1)
    else:
        scores.append(0.0)
        valid.append(0)


if use_calibration:
    scores_array = np.array(scores)
    valid_array =  np.array(valid)
    indices, = np.where(valid_array[:] > 0.)
    the_scores = scores_array[indices].reshape(-1, 1)
    scores =log_reg.predict_proba( the_scores)[:,1]
    print(scores)
    scores_array[indices] = scores
else:
    scores_array = np.array(scores)
    valid_array =  np.array(valid)

with open(scores_file, "w") as _file:
    for i in range(np.shape(scores_array)[0]):
        _file.write(str(scores_array[i])+'\t'+str(valid_array[i])+'\n')

#



