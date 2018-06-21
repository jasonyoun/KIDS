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
from sklearn.isotonic import IsotonicRegression as IR
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.linear_model import LogisticRegression
import configparser

parser = argparse.ArgumentParser(description='build stacked ensemble')

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
USE_SMOTE_SAMPLING=config.getboolean('DEFAULT','USE_SMOTE_SAMPLING')
LOG_REG_CALIBRATE= config.getboolean('DEFAULT','LOG_REG_CALIBRATE')


fn = open(model_save_dir+'/model.pkl','rb')





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



model_dic = pickle.load(fn)
pred_dic,dev_x,dev_y,predicates_dev = features.get_x_y('dev',er_mlp_model_dir,pra_model_dir)
dev_y[:][dev_y[:] == -1] = 0
predictions_dev = np.zeros_like(predicates_dev)
predictions_dev = predictions_dev.reshape((np.shape(predictions_dev)[0],1))
best_thresholds = np.zeros(len(pred_dic));
clf_dic = {}
for k,i in pred_dic.items():
    indices, = np.where(predicates_dev == i)
    if np.shape(indices)[0]!=0:
        model = model_dic[i]
        X = dev_x[indices]
        y = dev_y[indices]
        preds_before_sampling = model.predict_proba(X)[:,1]
        preds_before_sampling= preds_before_sampling.reshape((np.shape(preds_before_sampling)[0],1))
        preds = preds_before_sampling[:]
        if USE_SMOTE_SAMPLING:
            ros = SMOTE(ratio='minority')
            X_train, y_train = ros.fit_sample(preds, y.ravel() )
        else:
            X_train = preds
            y_train = y
        if LOG_REG_CALIBRATE:
            clf = LogisticRegression()
            clf.fit( X_train, y_train.ravel() )  
            preds = clf.predict_proba(X_train)[:,1]
            preds_before_sampling = clf.predict_proba(preds_before_sampling)[:,1]
        else:
            clf = IR(out_of_bounds='clip'  )
            clf.fit( X_train.ravel(), y_train.ravel()  )
            preds = clf.transform( X_train.ravel() )
            preds_before_sampling = clf.transform( preds_before_sampling.ravel() )
        clf_dic[i]=clf
        best_thresholds[i] = determine_thresholds(preds_before_sampling,y )







threshold = best_thresholds
print(threshold)

with open(model_save_dir+'/calibrations.pkl', 'wb') as output:
    pickle.dump(clf_dic, output, pickle.HIGHEST_PROTOCOL)

with open(model_save_dir+'/calibrated_threshold.pkl', 'wb') as output:
    pickle.dump(best_thresholds, output, pickle.HIGHEST_PROTOCOL)

print('thresholds saved in: '+model_save_dir)



    


