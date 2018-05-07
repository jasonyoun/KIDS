import argparse
import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score,confusion_matrix
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
plot_roc(len(pred_dic), test_y, probabilities,predicates,pred_dic, args.dir)
plot_pr(len(pred_dic), test_y, probabilities,predicates,pred_dic, args.dir)
roc_auc_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)
pr_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)


fl_measure_test = f1_score(test_y, y_hat)
accuracy_test = accuracy_score(test_y, y_hat)
confusion_test = confusion_matrix(test_y, y_hat)
for i in range(len(pred_dic)):
    for key, value in pred_dic.items():
        if value == i:
            pred_name =key
    indices, = np.where(predicates == i)
    if np.shape(indices)[0]!=0:
        classifications_predicate = y_hat[indices]
        labels_predicate = test_y[indices]
        fl_measure_predicate= f1_score(labels_predicate, classifications_predicate)
        accuracy_predicate = accuracy_score(labels_predicate, classifications_predicate)
        confusion_predicate = confusion_matrix(labels_predicate, classifications_predicate)
        print(" - predicate f1 measure for "+pred_name+ ":"+ str(fl_measure_predicate))
        print(" - predicate accuracy for "+pred_name+ ":"+ str(accuracy_predicate))
        print(" - predicate confusion matrix for "+pred_name+ ":")
        print(confusion_predicate)
        print(" ")

print("test f1 measure:"+ str(fl_measure_test))
print("test accuracy:"+ str(accuracy_test))
print("test confusion matrix:")
print(confusion_test)

    


