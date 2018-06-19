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

# min_score = np.min(probabilities) 
# print(min_score)
# max_score = np.max(probabilities)
# print(max_score) 
# np.savetxt('dev_predictions.txt',probabilities)
# best_threshold = np.zeros(len(pred_dic));
# best_f1_metric= np.zeros(len(pred_dic));
# for i in range(len(pred_dic)):
#     best_threshold[i]= min_score;
#     best_f1_metric[i] = -1;

# score = min_score
# increment = 0.00001
# print(dev_y)
# while(score <= max_score):
#     for i in range(len(pred_dic)):
#         predicate_indices = np.where(predicates == i)[0]
#         if np.shape(predicate_indices)[0]!=0:
#             predicate_predictions = probabilities[predicate_indices]
#             # predicate_predictions = predicate_predictions.astype(float)
#             # predictions = (predicate_predictions >= score) * 2 -1
#             predictions = (predicate_predictions >= score)
#             predictions = predictions.astype(int)
#             predicate_labels = dev_y[predicate_indices]

#             result = accuracy_score(predictions,predicate_labels)

#             result = result.astype(float)
#             f1_metric = np.mean(result)
#             if f1_metric > best_f1_metric[i]:
#                 print('predicate')
#                 print(i)
#                 print(f1_metric)
#                 best_threshold[i] = score
#                 best_f1_metric[i] = f1_metric
#             score += increment

# min_score = np.min(probabilities) 
# print(min_score)
# max_score = np.max(probabilities)
# print(max_score) 
# np.savetxt('dev_predictions.txt',probabilities)
# best_threshold = 0
# best_f1_metric= 0


# score = min_score
# increment = 0.000001
# print(dev_y)
# while(score <= max_score):
#     probabilities
#     # predicate_predictions = predicate_predictions.astype(float)
#     # predictions = (predicate_predictions >= score) * 2 -1
#     predictions = (probabilities >= score)
#     predictions = predictions.astype(int)
#     predicate_labels = dev_y

#     f1_metric = f1_score(predictions,predicate_labels)

#     # result = result.astype(float)
#     # f1_metric = np.mean(result)
#     if f1_metric > best_f1_metric:
#         # print('predicate')
#         # print(i)
#         print(f1_metric)
#         best_threshold = score
#         best_f1_metric = f1_metric
#     score += increment

predictions_list = probabilities.reshape(-1,1)
dev_labels = dev_y.reshape(-1,1)
both = np.column_stack((predictions_list,dev_labels))
both = both[both[:,0].argsort()]
predictions_list = both[:,0].ravel()
dev_labels = both[:,1].ravel()
best_accuracy = -1
accuracies = np.zeros(np.shape(predictions_list))
for i in range(np.shape(predictions_list)[0]):
    score = predictions_list[i]
    predictions = (predictions_list >= score)
    # accuracy = accuracy_score(predictions, dev_labels)
    accuracy = f1_score(dev_labels,predictions)
    accuracies[i] = accuracy
indices=np.argmax(accuracies)
best_threshold = np.mean(predictions_list[indices])


print(best_threshold)


threshold = best_threshold

with open(args.dir+'/threshold.pkl', 'wb') as output:
    pickle.dump(best_threshold, output, pickle.HIGHEST_PROTOCOL)

print('thresholds saved in: '+args.dir)


    


