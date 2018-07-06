import argparse
import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score,make_scorer
from scipy import interp
from sklearn.model_selection import  RandomizedSearchCV
abs_path_metrics= os.path.join(directory, '../utils')
sys.path.insert(0, abs_path_metrics)
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats, accuracy_score
from sklearn.model_selection import PredefinedSplit
import features
import plot_reliabilities
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import configparser


parser = argparse.ArgumentParser(description='build stacked ensemble')
parser.add_argument('--pra', metavar='pra_model (pra_model_2)', nargs='?',action='store',required=True,
                    help='The pra models to add')
parser.add_argument('--er_mlp', metavar='er_mlp_model (er_mlp_model_2)', nargs='?', action='store',required=True,
                    help='The er-mlp models to add')
parser.add_argument('--dir', metavar='dir', nargs='?', action='store',required=True,
                    help='directory to store the model')

args = parser.parse_args()
print(args)
config = configparser.ConfigParser()
model_instance_dir='model_instance/'
model_save_dir=model_instance_dir+args.dir+'/'
configuration = model_save_dir+args.dir.replace('/','')+'.ini'
print('./'+configuration)
config.read('./'+configuration)

print('configuration: ')
RUN_RANDOM_SEARCH=config.getboolean('DEFAULT','RUN_RANDOM_SEARCH')
TRAIN_DIR = config['DEFAULT']['TRAIN_DIR']
TEST_DIR = config['DEFAULT']['TEST_DIR']
DEV_DIR = config['DEFAULT']['DEV_DIR']
MODEL_SAVE_DIRECTORY=model_save_dir
F1_FOR_THRESHOLD = config.getboolean('DEFAULT','F1_FOR_THRESHOLD')
USE_SMOTE_SAMPLING=config.getboolean('DEFAULT','USE_SMOTE_SAMPLING')
LOG_REG_CALIBRATE= config.getboolean('DEFAULT','LOG_REG_CALIBRATE')
RANDOM_SEARCH_ESTIMATORS = config.items('RANDOM_SEARCH_ESTIMATORS')
RS_E_START = config.getint('RANDOM_SEARCH_ESTIMATORS','START')
RS_E_END = config.getint('RANDOM_SEARCH_ESTIMATORS','END')
RS_E_INCREMENT = config.getint('RANDOM_SEARCH_ESTIMATORS','INCREMENT')

RS_LR_START = config.getfloat('RANDOM_SEARCH_LEARNING_RATE','START')
RS_LR_END = config.getfloat('RANDOM_SEARCH_LEARNING_RATE','END')
RS_LR_INCREMENT = config.getfloat('RANDOM_SEARCH_LEARNING_RATE','INCREMENT')
RANDOM_SEARCH_COUNT = config.getint('DEFAULT','RANDOM_SEARCH_COUNT')
RANDOM_SEARCH_PROCESSES = config.getint('DEFAULT','RANDOM_SEARCH_PROCESSES')


config['DEFAULT']['ER_MLP_MODEL_DIR']=args.er_mlp
config['DEFAULT']['PRA_MODEL_DIR']=args.pra


def perform_randomized_search(pred_dic,train_x,train_y,test_x,test_y):

    def get_results_of_search(results, count=5):
        # print(results)
        print("")
        print("")
        print("roc_auc")
        print("")
        for i in range(1, count + 1):
            runs = np.flatnonzero(results['rank_test_roc_auc'] == i)
            for run in runs:
                print("evaluation rank: "+str(i))
                print("score: " + str(results['mean_test_roc_auc'][run])) 
                print("std: "+str(results['std_test_roc_auc'][run]))
                print(results['params'][run])
                print("")
        print("")
        print("")
        print("f1")
        print("")
        for i in range(1, count + 1):
            runs = np.flatnonzero(results['rank_test_f1'] == i)
            for run in runs:
                print("evaluation rank: "+str(i))
                print("score: " + str(results['mean_test_f1'][run])) 
                print("std: "+str(results['std_test_f1'][run]))
                print(results['params'][run])
                print("")
        print("")
        print("")
        print("average precision")
        print("")
        for i in range(1, count + 1):
            runs = np.flatnonzero(results['rank_test_average_precision'] == i)
            for run in runs:
                print("evaluation rank: "+str(i))
                print("score: " + str(results['mean_test_average_precision'][run])) 
                print("std: "+str(results['std_test_average_precision'][run]))
                print(results['params'][run])
                print("")
                if i==1:
                    print(results['params'][run])
                    AP_params = results['params'][run]

        print("")
        print("")
        print("accuracy")
        print("")
        for i in range(1, count + 1):
            runs = np.flatnonzero(results['rank_test_accuracy'] == i)
            for run in runs:
                print("evaluation rank: "+str(i))
                print("score: " + str(results['mean_test_accuracy'][run])) 
                print("std: "+str(results['std_test_accuracy'][run]))
                print(results['params'][run])
                print("")
        return AP_params
    param_distribution = { 'learning_rate':  np.arange(float(RS_LR_START),float(RS_LR_END), float(RS_LR_INCREMENT)), 'n_estimators':np.arange(int(RS_E_START),int(RS_E_END), int(RS_E_INCREMENT))}
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=0)
    all_x = np.vstack((train_x,test_x))
    all_y = np.vstack((train_y,test_y))
    all_y = all_y.astype(int)
    train_indices = np.full(np.shape(train_x)[0],-1)
    test_indices = np.full(np.shape(test_x)[0],0)
    test_fold = np.hstack((train_indices,test_indices))
    ps = PredefinedSplit(test_fold)
    search_count = RANDOM_SEARCH_COUNT
    random_search = RandomizedSearchCV(clf, param_distributions=param_distribution,n_iter=search_count,n_jobs=RANDOM_SEARCH_PROCESSES,scoring=['roc_auc','f1','average_precision','accuracy'],cv=ps,refit='average_precision')
    random_search.fit(all_x,all_y.ravel())
    AP_params = get_results_of_search(random_search.cv_results_)
    return AP_params

pred_dic,train_x,train_y,predicates_train = features.get_x_y(TRAIN_DIR,args.er_mlp,args.pra)
pred_dic,test_x,test_y,predicates_test = features.get_x_y(DEV_DIR,args.er_mlp,args.pra)
model_dic = {}
predictions_test = np.zeros_like(predicates_test,dtype=float)
predictions_test = predictions_test.reshape((np.shape(predictions_test)[0],1))
for k,i in pred_dic.items():
    test_indices, = np.where(predicates_test == i)
    if np.shape(test_indices)[0]!=0:
        test_x_predicate = test_x[test_indices]
        test_y_predicate = test_y[test_indices]

        train_indices, = np.where(predicates_train == i)
        if np.shape(train_indices)[0]!=0:
            train_x_predicate = train_x[train_indices]
            train_y_predicate = train_y[train_indices]
        else:
            print('No training data for predicate: '+k)
            sys.exit()
        if USE_SMOTE_SAMPLING:
            ros = SMOTE(ratio='minority')
            train_x_predicate, train_y_predicate = ros.fit_sample(train_x_predicate, train_y_predicate.ravel() )
            train_y_predicate = train_y_predicate.reshape(-1, 1)
        if RUN_RANDOM_SEARCH:
            AP_params = perform_randomized_search(pred_dic,train_x_predicate,train_y_predicate,test_x_predicate,test_y_predicate)
            config['RANDOM_SEARCH_BEST_PARAMS'+'_'+k]={'n_estimators':AP_params['n_estimators'],'learning_rate':AP_params['learning_rate'] }
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                n_estimators=config.getint('RANDOM_SEARCH_BEST_PARAMS'+'_'+k,'n_estimators'),learning_rate=config.getfloat('RANDOM_SEARCH_BEST_PARAMS'+'_'+k,'learning_rate'),random_state=0)
        clf.fit(train_x_predicate, train_y_predicate.ravel())
        preds = clf.predict_proba(test_x_predicate)[:,1]
        preds= preds.reshape((np.shape(preds)[0],1))
        predictions_test[test_indices] = preds[:]
        model_dic[i] = clf

score = pr_stats(len(pred_dic), test_y, predictions_test,predicates_test,pred_dic)
roc_ = roc_auc_stats(len(pred_dic), test_y, predictions_test,predicates_test,pred_dic)
print('score: '+ str(score))
print('roc_: '+ str(roc_))

with open(model_save_dir+'/model.pkl', 'wb') as output:
    pickle.dump(model_dic, output, pickle.HIGHEST_PROTOCOL)

with open('./'+configuration, 'w') as configfile: config.write(configfile)






   


