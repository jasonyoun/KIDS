import argparse
import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from scipy import interp
abs_path_metrics= os.path.join(directory, '../utils')
sys.path.insert(0, abs_path_metrics)
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
import features

parser = argparse.ArgumentParser(description='build stacked ensemble')
parser.add_argument('--pra', metavar='pra_model (pra_model_2)', nargs='+',action='store',required=True,
                    help='The pra models to add')
parser.add_argument('--er_mlp', metavar='er_mlp_model (er_mlp_model_2)', nargs='+', action='store',required=True,
                    help='The er-mlp models to add')
parser.add_argument('--dir', metavar='dir', nargs='?', action='store',required=True,
                    help='directory to store the model')

args = parser.parse_args()
print(args)



fn = open(args.dir+'/model.pkl','rb')
clf = pickle.load(fn)
pred_dic,test_x,test_y,predicates = features.get_x_y('test',args.er_mlp,args.pra)

y_hat = clf.predict(test_x)
probabilities = clf.predict_proba( test_x)
print(probabilities)
probabilities = probabilities[:,1]
# print(y_hat)
print(probabilities)
print(average_precision_score(test_y,probabilities))


with open(args.dir+"/predictions.txt", 'w') as _file:
    for i in range(np.shape(y_hat)[0]):
        _file.write("predicate: "+str(predicates[i])+"\tclassification: "+str(int(y_hat[i]))+ '\tprediction: '+str(probabilities[i])+'\tlabel: '+str(int(test_y[i]))+'\n' )

    


