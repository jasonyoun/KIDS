import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv

DATA_PATH=sys.argv[1]
with open(DATA_PATH+'/selected_relations') as f:
    relations = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
relations = [x.strip() for x in relations] 
_relations = []
for r in relations:
    _relations.append('_'+r)

relations +=_relations

for r in relations:
    pos_df = pd.read_csv(DATA_PATH+'queriesR_train/'+r,sep='\t',encoding ='latin-1', 
                  names = ["subject", "object"])
    neg_df = pd.read_csv(DATA_PATH+'queriesR_train_neg/'+r,sep='\t',encoding ='latin-1', 
                  names = ["subject", "object"])
    result = pd.merge(pos_df, neg_df, how='outer', on="subject")
    result.to_csv(DATA_PATH+'queriesR_train/'+r, sep='\t', index=False,header=False)


