import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score,confusion_matrix
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
import argparse

parser = argparse.ArgumentParser(description='evaluate the results')
parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')

args = parser.parse_args()
print(args.dir)


with open('./selected_relations') as f:
    relations = f.readlines()
relations = [x.strip() for x in relations] 
print(relations)

index=0
predicates_dic = {}
for r in relations:
    predicates_dic[r] =  index
    index+=1

combined_scores_array = None
combined_predicates_array = None
combined_labels_array = None
combined_classifications_array = None
start = 0
for k,v in predicates_dic.items():
    with open(args.dir+'/scores/'+k, "r") as _file, open(args.dir+'/queriesR_labels/'+k, "r") as l_file, open(args.dir+'/classifications/'+k, "r") as c_file:
        scores = _file.readlines()
        scores = [x.strip().split('\t')[0] for x in scores] 
        labels = l_file.readlines()
        labels = [x.strip().split('\t')[2] for x in labels] 
        classifications = c_file.readlines()

        classifications = [x.strip().split('\t')[0] for x in classifications] 

        predicates = [v for x in scores] 
        predicates_array = np.array(predicates)
        scores_array = np.array(scores)
        labels_array = np.array(labels)
        classifications_array = np.array(classifications)
        if start ==0:
            combined_scores_array = scores_array
            combined_predicates_array = predicates_array
            combined_labels_array = labels_array
            combined_classifications_array = classifications_array
            start+=1
        else:
            combined_scores_array = np.append(combined_scores_array,scores_array)
            combined_predicates_array = np.append(combined_predicates_array,predicates_array)
            combined_labels_array = np.append(combined_labels_array,labels_array)
            combined_classifications_array = np.append(combined_classifications_array,classifications_array)

combined_scores_array = np.transpose(combined_scores_array).astype(float)
combined_predicates_array = np.transpose(combined_predicates_array).astype(int)
combined_labels_array = np.transpose(combined_labels_array).astype(int)
combined_classifications_array = np.transpose(combined_classifications_array).astype(int)
combined_labels_array[:][combined_labels_array[:] == -1] = 0
for i in range(len(predicates_dic)):
    for key, value in predicates_dic.items():
        if value == i:
            pred_name =key
    indices, = np.where(combined_predicates_array == i)
    classifications_predicate = combined_classifications_array[indices]
    labels_predicate = combined_labels_array[indices]
    classifications_predicate[:][classifications_predicate[:] == -1] = 0
    fl_measure_predicate = f1_score(labels_predicate, classifications_predicate)
    accuracy_predicate = accuracy_score(labels_predicate, classifications_predicate)
    confusion_predicate = confusion_matrix(labels_predicate, classifications_predicate)
    print(" - test f1 measure for "+pred_name+ ":"+ str(fl_measure_predicate))
    print(" - test accuracy for "+pred_name+ ":"+ str(accuracy_predicate))
    print(" - test confusion matrix for "+pred_name+ ":")
    print(confusion_predicate)
    print(" ")

mean_average_precision_test = pr_stats(len(relations), combined_labels_array, combined_scores_array,combined_predicates_array,predicates_dic)
roc_auc_test = roc_auc_stats(len(relations), combined_labels_array, combined_scores_array,combined_predicates_array,predicates_dic)
fl_measure_test = f1_score(combined_labels_array, combined_classifications_array)
accuracy_test = accuracy_score(combined_labels_array, combined_classifications_array)
plot_pr(len(relations), combined_labels_array, combined_scores_array,combined_predicates_array,predicates_dic, args.dir+'/')
plot_roc(len(relations), combined_labels_array, combined_scores_array,combined_predicates_array,predicates_dic, args.dir+'/')

print("test mean average precision:"+ str(mean_average_precision_test))
print("test f1 measure:"+ str(fl_measure_test))
print("test accuracy:"+ str(accuracy_test))
print("test roc auc:"+ str(roc_auc_test))










