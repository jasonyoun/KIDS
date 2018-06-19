import argparse
import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from scipy import interp





ER_MLP_MODEL_HOME='../er_mlp/model/'
PRA_MODEL_HOME='../pra/model/'
def get_x_y(which,er_mlp,pra):

    er_mlp_features_array_list=[]
    for i in er_mlp:
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

    pra_features_array_list = []
    for i in pra:
        model_base_dir = PRA_MODEL_HOME+'/'+i+'/instance/'

        pra_features = []
        for k,v in pred_dic.items():
            if os.path.isfile(model_base_dir+'/'+which+'/scores/'+k):
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
    # print(p_features)
    predicates_pra= p_features[:,0]

    predicates_pra = predicates_pra.astype(int)
    predicates_er_mlp= e_features[:,0]
    predicates_er_mlp = predicates_er_mlp.astype(int)
    p_predicate_indices = np.where(predicates_pra[:] == 0)[0]
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
        # print(e_features[e_predicate_indices][:,1:])
        predicates_list.append(predicates_er_mlp[e_predicate_indices])
        combined_list.append(np.hstack((e_features[e_predicate_indices][:,1:],p_features[p_predicate_indices][:,1:])))

    combined_array = np.vstack(combined_list)
    y = np.vstack(labels_list)
    predicates = np.hstack(predicates_list)
    y[:][y[:] == -1] = 0
    # np.savetxt(which+'_blah.txt',combined_array)
    return pred_dic,combined_array[:,[0,2,3]],y,predicates

    


