import argparse
import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score,confusion_matrix, precision_score, recall_score
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
parser.add_argument('--use_calibration',action='store_const',default=False,const=True)

args = parser.parse_args()
print(args)
use_calibration = args.use_calibration
if use_calibration:
    with open(args.dir+'/calibrations.pkl', 'rb') as pickle_file:
        log_reg = pickle.load(pickle_file)

# def classify( predictions_list,threshold, predicates):
#     classifications = []
#     for i in range(len(predictions_list)):
#         # print(predictions_list[i])
#         # print(threshold[predicates[i]])
#         if(predictions_list[i] >= threshold[predicates[i]]):
#             classifications.append(1)
#         else:
#             classifications.append(0)
#     return np.array(classifications)
def classify( predictions_list,threshold, predicates):
    classifications = []
    for i in range(len(predictions_list)):
        # print(predictions_list[i])
        # print(threshold[predicates[i]])
        if(predictions_list[i] >= threshold):
            classifications.append(1)
        else:
            classifications.append(0)
    return np.array(classifications)

fn = open(args.dir+'/threshold.pkl','rb')
if use_calibration:
    fn = open(args.dir+'/calibrated_threshold.pkl','rb')
threshold = pickle.load(fn)

print(threshold)

fn = open(args.dir+'/model.pkl','rb')
clf = pickle.load(fn)
pred_dic,test_x,test_y,predicates = features.get_x_y('test',args.er_mlp,args.pra)
# test_y[:][test_y[:] == -1] = 0
print(test_y)
# y_hat = clf.predict(test_x)
probabilities = clf.predict_proba( test_x)
# print(probabilities)
probabilities = probabilities[:,1]
if use_calibration:
    # scores=log_reg.transform( probabilities.ravel() )
    scores =log_reg.predict_proba( probabilities.reshape(-1,1))[:,1]
    probabilities = scores

# np.set_printoptions(threshold=np.nan)
print('probabilities')
print(probabilities)
print('test_y')
print(test_y)

classifications = clf.predict( test_x)
# classifications = classify(probabilities,threshold,predicates)
plot_roc(len(pred_dic), test_y, probabilities,predicates,pred_dic, args.dir)
plot_pr(len(pred_dic), test_y, probabilities,predicates,pred_dic, args.dir)
auc_test = roc_auc_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)
ap_test = pr_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)


fl_measure_test = f1_score(test_y, classifications)
accuracy_test = accuracy_score(test_y, classifications)
recall_test = recall_score(test_y, classifications)
precision_test = precision_score(test_y, classifications)
confusion_test = confusion_matrix(test_y, classifications)
for i in range(len(pred_dic)):
    for key, value in pred_dic.items():
        if value == i:
            pred_name =key
    indices, = np.where(predicates == i)
    if np.shape(indices)[0]!=0:
        print(indices)
        print(classifications)
        classifications_predicate = classifications[indices]
        labels_predicate = test_y[indices]
        fl_measure_predicate= f1_score(labels_predicate, classifications_predicate)
        accuracy_predicate = accuracy_score(labels_predicate, classifications_predicate)
        recall_predicate= recall_score(labels_predicate, classifications_predicate)
        precision_predicate = precision_score(labels_predicate, classifications_predicate)
        confusion_predicate = confusion_matrix(labels_predicate, classifications_predicate)
        print(" - predicate f1 measure for "+pred_name+ ":"+ str(fl_measure_predicate))
        print(" - predicate accuracy for "+pred_name+ ":"+ str(accuracy_predicate))
        print(" - predicate recall for "+pred_name+ ":"+ str(recall_predicate))
        print(" - predicate precision for "+pred_name+ ":"+ str(precision_predicate))
        print(" - predicate confusion matrix for "+pred_name+ ":")
        print(confusion_predicate)
        print(" ")

_file =  args.dir+"/classifications_stacked.txt"
with open(_file, 'w') as t_f:
    for row in classifications:
        t_f.write(str(row)+'\n')


print("test f1 measure:"+ str(fl_measure_test))
print("test accuracy:"+ str(accuracy_test))
print("test precision:"+ str(precision_test))
print("test recall:"+ str(recall_test))
print("test confusion matrix:")
print('test auc: '+str(auc_test))
print('test ap: '+str(ap_test))
print(confusion_test)

print(ap_test)

    


