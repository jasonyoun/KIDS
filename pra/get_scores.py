import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv
relation = sys.argv[1]
queries_file = 'queriesR_test/'+relation
if len(sys.argv)>2:
    queries_file = 'queriesR_'+sys.argv[2]+'/'+relation
fname='predictions/'+relation

scores_file = 'scores/'+relation
with open(fname) as f:
    lines = f.readlines()

with open(queries_file, "r") as l_file:
    queries = l_file.readlines()

with open(scores_file, "w") as _file:
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
            _file.write(str(float(score_and_entity[0]))+'\t1\n')
        else:
            _file.write(str(0.0)+'\t0\n')



