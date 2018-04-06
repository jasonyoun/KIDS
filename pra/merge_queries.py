import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv
import os.path

DATA_PATH=sys.argv[1]
with open(DATA_PATH+'/selected_relations') as f:
    relations = f.readlines()
relations = [x.strip() for x in relations] 
_relations = []
for r in relations:
    _relations.append('_'+r)

relations +=_relations

for r in relations:
    pos_df = pd.read_csv(DATA_PATH+'queriesR_train/'+r,sep='\t',encoding ='latin-1', 
                  names = ["subject", "object"])
    if not os.path.isfile(DATA_PATH+'queriesR_train_neg/'+r):
    	continue
    neg_df = pd.read_csv(DATA_PATH+'queriesR_train_neg/'+r,sep='\t',encoding ='latin-1', 
                  names = ["subject", "object"])
    result = pd.merge(pos_df, neg_df, how='left', on="subject")
    result.to_csv(DATA_PATH+'queriesR_train/'+r, sep='\t', index=False,header=False)


