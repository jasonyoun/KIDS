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

parser = argparse.ArgumentParser(description='generate test queries')
parser.add_argument('--predicate', nargs='?',required=True,
                    help='the predicate that we will get the scores for')
parser.add_argument('--data_file', metavar='dir', nargs='?', default='dev.txt',
                    help='base directory')

parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')

args = parser.parse_args()
print(args.dir)

relation = args.predicate


data_file = args.data_file



with open(data_file, "r") as _file:
    lines = _file.readlines()

queries = {}
queries_neg = {}
queries_copy = {}
for line in lines:
    columns = line.strip().split('\t')
    columns = [x.strip() for x in columns]
    if columns[1]==relation:
        subject = "c$"+columns[0]
        subject_copy = "c$"+columns[0]
        _object =  "c$"+columns[2]
        _object_copy =  "c$"+columns[2]
        if columns[3]=='1':
            if subject not in queries:
                queries[subject] = set()
                # queries_copy[subject_copy] = set()
                queries[subject].add(_object)
                # queries_copy[subject].add(_object_copy)
            else:
                queries[subject].add(_object)
                # queries_copy[subject_copy].add(_object_copy)
        else:
            if subject not in queries_neg:
                queries_neg[subject] = set()
                # queries_copy[subject_copy] = set()
                queries_neg[subject].add(_object)
                # queries_copy[subject_copy].add(_object_copy)
            else:
                queries_neg[subject].add(_object)
                # queries_copy[subject_copy].add(_object_copy)






with open(args.dir+"/queriesR_test/"+relation, "w") as _file:
    for k,v in queries.items():
        # v_copy = type(v)(v) 
        o = random.sample(v,1)[0]
        _file.write(k+'\t'+o)
        v.remove(o)
        for o in v:
            _file.write(' '+o)
        _file.write('\t')
        if k in queries_neg:
            neg_v = queries_neg[k]
            o = random.sample(neg_v,1)[0]
            _file.write(o)
            neg_v.remove(o)
            for o in neg_v:
                _file.write(' '+o)
        _file.write('\n')
    for k,v in queries_neg.items():
        if k not in queries:
            o = random.sample(v,1)[0]
            _file.write(k+'\t\t'+o)
            v.remove(o)
            for o in v:
                _file.write(' '+o)
            _file.write('\n')


    # few_more=15
    # count=0
    # for k,v in queries_copy.items():
    #     v_copy = type(v)(v) 
    #     _file.write(k+'\t'+random.sample(v,1)[0])
    #     for o in v:
    #         _file.write(' '+o)
    #     _file.write('\n')
    #     if count < few_more:
    #         count+=1
    #     else:
    #         break

        # v_copy_copy = type(v_copy)(v_copy) 
        # _file.write(k+'\t'+v_copy.pop())
        # for o in v_copy:
        #     _file.write(' '+o)
        # _file.write('\n')

        # _file.write(k+'\t'+v_copy_copy.pop())
        # for o in v_copy_copy:
        #     _file.write(' '+o)
        # _file.write('\n')





