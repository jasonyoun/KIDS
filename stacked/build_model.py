import argparse
import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score,make_scorer
from scipy import interp
from sklearn.model_selection import  RandomizedSearchCV
abs_path_metrics= os.path.join(directory, '../utils')
sys.path.insert(0, abs_path_metrics)
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
from sklearn.model_selection import PredefinedSplit
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

ER_MLP_MODEL_HOME='../er_mlp/model/'
PRA_MODEL_HOME='../pra/model/'

def perform_grid_search(pred_dic,train_x,train_y,test_x,test_y):
    #{'learning_rate': 0.027799999999999984, 'max_depth': 3, 'n_estimators': 100}
    param_grid = { 'learning_rate':  np.arange(0.1, 2.0, 0.2),'n_estimators': np.arange(20, 300, 50)}
    grid = list(ParameterGrid(param_grid))
    for i in grid:
        print(i)
        # clf = GradientBoostingClassifier(n_estimators=i['n_estimators'], learning_rate=i['learning_rate'],max_depth=i['max_depth'], random_state=0)
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=i['n_estimators'],
                             learning_rate=i['learning_rate'], random_state=0)

        #AdaBoostClassifier(n_estimators=i['n_estimators'], learning_rate=i['learning_rate'],max_depth=1, random_state=0)
        clf.fit(train_x, train_y.ravel())
        probabilities = clf.predict_proba( test_x)[:,1]
        score = pr_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)
        roc_ = roc_auc_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)
        print('score: '+ str(score))
        print('roc_: '+ str(roc_))
        if max_score > score:
            max_score = score
            best_params = i

def perform_randomized_search(pred_dic,train_x,train_y,test_x,test_y):

    param_grid = { 'learning_rate':  np.arange(0.1, 2.0, 0.2),'n_estimators': np.arange(20, 300, 50)}
    # param_grid = { 'alpha': np.arange(1.0, 2.0, 0.1),}

    grid = list(ParameterGrid(param_grid))
    max_score = 0

    def get_results_of_search(results, count=40):
        print(results)
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

    param_distribution = { 'learning_rate':  np.arange(0.1, 2.0, 0.1),'n_estimators': np.arange(20, 700, 50)}

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=0)
    all_x = np.vstack((train_x,test_x))
    all_y = np.vstack((train_y,test_y))
    train_indices = np.full(np.shape(train_x)[0],-1)
    test_indices = np.full(np.shape(test_x)[0],0)
    test_fold = np.hstack((train_indices,test_indices))
    ps = PredefinedSplit(test_fold)
    search_count = 40
    random_search = RandomizedSearchCV(clf, param_distributions=param_distribution,n_iter=search_count,n_jobs=4,scoring=['average_precision','f1'],cv=ps,refit='f1')
    random_search.fit(all_x,all_y.ravel())
    get_results_of_search(random_search.cv_results_)

pred_dic,train_x,train_y,predicates = features.get_x_y('predictions',args.er_mlp,args.pra)
pred_dic,test_x,test_y,predicates = features.get_x_y('test',args.er_mlp,args.pra)

perform_randomized_search(pred_dic,train_x,train_y,test_x,test_y)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=120,learning_rate=1.2, random_state=0)
#clf = GradientBoostingClassifier(n_estimators=106, learning_rate=0.027799999999999984,max_depth=3, random_state=0)
clf.fit(train_x, train_y.ravel())

with open(args.dir+'/model.pkl', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)



    


