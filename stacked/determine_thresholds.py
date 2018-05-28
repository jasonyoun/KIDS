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
# print(args)



fn = open(args.dir+'/model.pkl','rb')
clf = pickle.load(fn)
pred_dic,dev_x,dev_y,predicates = features.get_x_y('dev',args.er_mlp,args.pra)
# dev_y[:][dev_y[:] == -1] = 0

y_hat = clf.predict(dev_x)
probabilities = clf.predict_proba( dev_x)
# print(probabilities)
probabilities = probabilities[:,1]

# for i in range(len(pred_dic)):
#     for key, value in pred_dic.items():
#         if value == i:
#             pred_name =key
    # indices, = np.where(predicates == i)
    # if np.shape(indices)[0]!=0:
    #     prob_predicate = probabilities[indices]
    #     labels_predicate = dev_y[indices]

min_score = np.min(probabilities) 
max_score = np.max(probabilities) 
np.savetxt('dev_predictions.txt',probabilities)
best_threshold = np.zeros(len(pred_dic));
best_f1_metric= np.zeros(len(pred_dic));
for i in range(len(pred_dic)):
    best_threshold[i]= min_score;
    best_f1_metric[i] = -1;

score = min_score
increment = 0.00001
while(score <= max_score):
    for i in range(len(pred_dic)):
        predicate_indices = np.where(predicates == i)[0]
        if np.shape(predicate_indices)[0]!=0:
            predicate_predictions = probabilities[predicate_indices]
            # predictions = (predicate_predictions >= score) * 2 -1
            predictions = (predicate_predictions >= score)
            predicate_labels = dev_y[predicate_indices]
            f1_metric = f1_score(predicate_labels,predictions)
            # print(predicate_labels)
            # accuracy = np.mean(predictions == predicate_labels)
            if f1_metric > best_f1_metric[i]:
                print(f1_metric)
                best_threshold[i] = score
                best_f1_metric[i] = f1_metric
            score += increment

print(best_threshold)


threshold = best_threshold

with open(args.dir+'/threshold.pkl', 'wb') as output:
    pickle.dump(best_threshold, output, pickle.HIGHEST_PROTOCOL)

print('thresholds saved in: '+args.dir)


    


