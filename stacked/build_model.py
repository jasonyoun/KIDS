import argparse
import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
from scipy import interp

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


def get_roc(num_preds, Y, predictions,predicates,pred_dic, directory):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    predicates_included = []
    for i in range(num_preds):
        predicate_indices = np.where(predicates == i)[0]
        if np.shape(predicate_indices)[0] == 0:
            continue
        else:
            predicates_included.append(i)
        predicate_predictions = predictions[predicate_indices]
        predicate_labels = Y[predicate_indices]

        fpr[i], tpr[i] , _ = roc_curve(predicate_labels.ravel(), predicate_predictions.ravel())
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in predicates_included]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in predicates_included:
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(predicates_included)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc["macro"]

# print(e_features)
# print(labels)
def get_x_y(which):

    er_mlp_features_array_list=[]
    for i in args.er_mlp:
        model_base_dir = ER_MLP_MODEL_HOME+i
        fn = open(model_base_dir+'/params.pkl','rb')
        params = pickle.load(fn)
        pred_dic = params['pred_dic']
        pred_index_dic = {}
        for k,v in pred_dic.items():
            pred_index_dic[v] = k

        with open(model_base_dir+'/'+which+'/predictions.txt', "r") as _file:
            predictions = _file.readlines()
        predictions = [x.strip() for x in predictions] 

        # print(predictions)
        er_mlp_features = []
        labels = []
        for line in predictions:
            # print(line)
            strings =  line.split('\t')
            predicate = int(strings[0].replace('predicate: ',''))
            pred = float(strings[2].replace('prediction: ',''))
            label = int(strings[3].replace('label: ',''))
            labels.append([label])
            er_mlp_features.append([predicate,pred,1])


        er_mlp_features_array = np.array(er_mlp_features)
        er_mlp_features_array_list.append(er_mlp_features_array)

    if len(er_mlp_features_array)>1:
        e_features = np.vstack(er_mlp_features_array)
    else:
        e_features = er_mlp_features_array

    labels_array = np.array(labels)

    print(np.shape(e_features))
    pra_features_array_list = []
    for i in args.pra:
        model_base_dir = PRA_MODEL_HOME+'/'+i+'/instance/'

        pra_features = []
        for k,v in pred_dic.items():
            with open(model_base_dir+'/'+which+'/scores/'+k, "r") as _file:
                predictions = _file.readlines()
                predictions = [x.strip() for x in predictions] 

                # print(predictions)
                for line in predictions:
                    # print(line)
                    strings =  line.split('\t')
                    pred = float(strings[0].strip())
                    valid = int(strings[1].strip())
                    pra_features.append([int(v),pred,valid])

        pra_features_array = np.array(pra_features)
        pra_features_array_list.append(pra_features_array)

    if len(pra_features_array_list)>1:
        p_features = np.vstack(pra_features_array_list)
    else:
        p_features = pra_features_array_list

    p_features = np.squeeze(p_features)
    # print(np.shape(p_features))

    predicates_pra= p_features[:,0]

    predicates_pra = predicates_pra.astype(int)
    predicates_er_mlp= e_features[:,0]
    predicates_er_mlp = predicates_er_mlp.astype(int)
    print(predicates_pra)
    p_predicate_indices = np.where(predicates_pra[:] == 0)[0]
    print(p_predicate_indices)
    # print(predicates_er_mlp)
    combined_list = []
    labels_list = []
    predicates_list = []
    for k,v in pred_dic.items():
        # print(k)
        # print(v)
        p_predicate_indices = np.where(predicates_pra[:] == v)[0]
        # print(p_predicate_indices)
        e_predicate_indices = np.where(predicates_er_mlp[:] == v)[0]
        labels_list.append(labels_array[e_predicate_indices])
        # print(predicates_er_mlp)
        predicates_list.append(predicates_er_mlp[e_predicate_indices])
        combined_list.append(np.hstack((e_features[e_predicate_indices][:,:],p_features[p_predicate_indices][:,:])))

    combined_array = np.vstack(combined_list)
    y = np.vstack(labels_list)
    print(predicates_list)
    predicates = np.hstack(predicates_list)
    return pred_dic,combined_array,y,predicates

# print(np.shape(combined_array))
# print(np.shape(y) )
#'learning_rate': np.arange(0.1, 1.0, 0.1),
param_grid = { 'max_depth' : np.arange(1, 5, 1),'learning_rate': np.arange(0.1, 2.0, 0.2),'n_estimators': np.arange(50, 250, 50)}
# param_grid = { 'alpha': np.arange(1.0, 2.0, 0.1),}

pred_dic,train_x,train_y,predicates = get_x_y('predictions')
pred_dic,test_x,test_y,predicates = get_x_y('test')
# grid = list(ParameterGrid(param_grid))
# max_score = 0
# for i in grid:
#     print(i)
#     clf = GradientBoostingClassifier(n_estimators=i['n_estimators'], learning_rate=i['learning_rate'],max_depth=i['max_depth'], random_state=0)
#     clf.fit(train_x, train_y.ravel())
#     probabilities = clf.predict_proba( test_x)[:,1]
#     score = get_roc(len(pred_dic), test_y, probabilities,predicates,pred_dic, './')
#     print('score: '+ str(score))
#     if max_score > score:
#         max_score = score
#         best_params = i

# print('best')
# print(best_params)
# print(score)
# score: 0.889247299361
# {'learning_rate': 1.3000000000000003, 'max_depth': 2, 'n_estimators': 150}
# clf = LogisticRegression()
clf = GradientBoostingClassifier(n_estimators=150, learning_rate=1.3,max_depth=2, random_state=0)

clf.fit(train_x, train_y.ravel())

with open(args.dir+'/model.pkl', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)



    


