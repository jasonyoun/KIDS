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
import configparser


parser = argparse.ArgumentParser(description='determine thresholds')
parser.add_argument('--dir', metavar='dir', nargs='?', action='store',required=True,
                    help='directory to store the model')
args = parser.parse_args()


config = configparser.ConfigParser()
model_instance_dir='model_instance/'
model_save_dir=model_instance_dir+args.dir+'/'
configuration = model_save_dir+args.dir.replace('/','')+'.ini'
print('./'+configuration)
config.read('./'+configuration)

er_mlp_model_dir = config['DEFAULT']['er_mlp_model_dir']
pra_model_dir = config['DEFAULT']['pra_model_dir']
F1_FOR_THRESHOLD = config.getboolean('DEFAULT','F1_FOR_THRESHOLD')

# print(args)

fn = open(model_save_dir+'/model.pkl','rb')
model_dic = pickle.load(fn)

def determine_thresholds(predictions_dev,dev_y):
    predictions_list = predictions_dev.reshape(-1,1)
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
        accuracy = accuracy_score(predictions, dev_labels)
        if F1_FOR_THRESHOLD:
            accuracy = f1_score(dev_labels,predictions)
        accuracies[i] = accuracy
    indices=np.argmax(accuracies)
    best_threshold = np.mean(predictions_list[indices])
    return best_threshold

pred_dic,dev_x,dev_y,predicates_dev = features.get_x_y('dev',er_mlp_model_dir,pra_model_dir)
# dev_y[:][dev_y[:] == -1] = 0
predictions_dev = np.zeros_like(predicates_dev,dtype=float)
predictions_dev = predictions_dev.reshape((np.shape(predictions_dev)[0],1))
best_thresholds = np.zeros(len(pred_dic));
for k,i in pred_dic.items():
    indices, = np.where(predicates_dev == i)
    if np.shape(indices)[0]!=0:
        clf = model_dic[i]
        X = dev_x[indices]
        y = dev_y[indices]
        preds = clf.predict_proba(X)[:,1]
        preds= preds.reshape((np.shape(preds)[0],1))
        best_thresholds[i] = determine_thresholds(preds,y)





print(best_thresholds)


threshold = best_thresholds

with open(model_save_dir+'/threshold.pkl', 'wb') as output:
    pickle.dump(best_thresholds, output, pickle.HIGHEST_PROTOCOL)

print('thresholds saved in: '+model_save_dir)


    


