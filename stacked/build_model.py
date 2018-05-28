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
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
from sklearn.model_selection import PredefinedSplit
import features
import plot_reliabilities
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.linear_model import LogisticRegression


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
    # param_grid = { 'n_estimators':  np.arange(1.98630000000002 ,1.9867, .00001)}
    param_grid = { 'learning_rate':  np.arange(1.98653 ,1.9866, .00001)}
    # param_grid = { 'C':  np.arange(.1,5, .1)}
    # 'learning_rate':  np.arange(.8,2.0, 0.01)
    #param_grid = { 'n_estimators': np.arange(1, 600, 1)}
    #207
    grid = list(ParameterGrid(param_grid))
    max_score=0
    best_params = None
    for i in grid:
        print(i)
        # clf = GradientBoostingClassifier(n_estimators=i['n_estimators'], learning_rate=i['learning_rate'],max_depth=i['max_depth'], random_state=0)
        # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
        #                      algorithm="SAMME",
        #                      n_estimators=i['n_estimators'],
        #                      learning_rate=i['learning_rate'], random_state=0)
        # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),learning_rate=i['n_estimators'] ,
        #                    n_estimators=207, random_state=0)
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             n_estimators=206, random_state=0,learning_rate=i['learning_rate'])
        #clf = RandomForestClassifier(max_depth=None,n_estimators=i['n_estimators'], random_state=0)
        # clf =  linear_model.LogisticRegression(C=i['C'])
        # AdaBoostClassifier(n_estimators=i['n_estimators'], learning_rate=i['learning_rate'],max_depth=1, random_state=0)
        print(train_x)
        print(train_y)
        print(test_x)
        print(test_y)
        clf.fit(train_x, train_y.ravel())
        probabilities = clf.predict_proba( test_x)[:,1]
        print(probabilities)
        score = pr_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)
        roc_ = roc_auc_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)
        print('score: '+ str(score))
        print('roc_: '+ str(roc_))
        if max_score > roc_:
            max_score = roc_
            best_params = i
    print(best_params)

def perform_randomized_search(pred_dic,train_x,train_y,test_x,test_y):


    def get_results_of_search(results, count=30):
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

# random 
# average precision

# evaluation rank: 1
# score: 0.4426102982826666
# std: 0.0
# {'n_estimators': 339, 'learning_rate': 1.885999999999985}
#just negative
# evaluation rank: 1
# score: 0.4163846122654644
# std: 0.0
# {'n_estimators': 647, 'learning_rate': 1.804999999999994}



    param_distribution = { 'learning_rate':  np.arange(1.0,2.0, 0.0001), 'n_estimators':np.arange(400,800, 1)}
    # param_distribution = {'n_estimators': np.arange(300, 800, 1)}
    #clf = GradientBoostingClassifier( random_state=0)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=0)
    # clf = RandomForestClassifier( random_state=0)
    all_x = np.vstack((train_x,test_x))
    all_y = np.vstack((train_y,test_y))
    all_y = all_y.astype(int)
    # np.savetxt('all_y.txt',all_y)
    # np.savetxt('all_x.txt',all_x)
    train_indices = np.full(np.shape(train_x)[0],-1)
    test_indices = np.full(np.shape(test_x)[0],0)
    test_fold = np.hstack((train_indices,test_indices))
    ps = PredefinedSplit(test_fold)
    search_count = 100
    random_search = RandomizedSearchCV(clf, param_distributions=param_distribution,n_iter=search_count,n_jobs=8,scoring=['roc_auc','f1','average_precision'],cv=ps,refit='average_precision')
    random_search.fit(all_x,all_y.ravel())
    get_results_of_search(random_search.cv_results_)
    return random_search

pred_dic,train_x,train_y,predicates = features.get_x_y('predictions',args.er_mlp,args.pra)
ros = SMOTE(ratio='minority')
train_x, train_y = ros.fit_sample(train_x, train_y.ravel() )
train_y = train_y.reshape(-1, 1)
print('shape')
print(np.shape(train_x))
print(train_x)
print(train_y)
pred_dic,test_x,test_y,predicates = features.get_x_y('test',args.er_mlp,args.pra)
print('shape')
print(np.shape(test_x))
print(test_y)

# random_search = perform_grid_search(pred_dic,train_x,train_y,test_x,test_y)
print('test_x')
# print(test_x.reshape(-1,1))
# test_x=test_x.reshape(-1,1)
# test_y[:][test_y[:] == -1] = 0

#perform_randomized_search(pred_dic,train_x,train_y,test_x,test_y)
# 'n_estimators': 20, 'learning_rate': 0.90000000000000002
#'n_estimators': 20, 'learning_rate': 1.9000000000000001
#'n_estimators': 336, 'learning_rate': 0.28000000000000003
# {'n_estimators': 343, 'learning_rate': 0.17000000000000001}
# {'n_estimators': 534, 'learning_rate': 1.2427999999999733}
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),learning_rate=1.2427999999999733,
                      n_estimators=534)
clf.fit(train_x, train_y.ravel())
probabilities = clf.predict_proba( test_x)[:,1]
score = pr_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)
roc_ = roc_auc_stats(len(pred_dic), test_y, probabilities,predicates,pred_dic)
print('score: '+ str(score))
print('roc_: '+ str(roc_))
#clf = GradientBoostingClassifier(n_estimators=106, learning_rate=0.027799999999999984,max_depth=3, random_state=0)
# clf.fit(train_x, train_y.ravel())

with open(args.dir+'/model.pkl', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)

# average precision
# evaluation rank: 1
# score: 0.40612947274574607
# std: 0.0
# {'n_estimators': 210, 'learning_rate': 1.9863000000000541}

# f1

# evaluation rank: 1
# score: 0.4129032258064516
# std: 0.0
# {'n_estimators': 210, 'learning_rate': 1.9863000000000541}

#Average precision
# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),learning_rate=1.986530000000002,
#                      n_estimators=206, random_state=0)


    


