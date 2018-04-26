import argparse
import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
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

def plot_roc(num_preds, Y, predictions,predicates,pred_dic, directory):
    baseline = np.zeros(np.shape(predictions)[0])
    baseline_fpr, baseline_tpr , _ = roc_curve(Y.ravel(), baseline.ravel())
    baseline_aucROC = auc(baseline_fpr, baseline_tpr)
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
        print(np.shape(fpr[i]))
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in predicates_included]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in predicates_included:
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(predicates_included)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"], lw=2, color='darkorange', label="Macro Average ROC curve (AUC:{:.3f})".format(roc_auc["macro"]))
    plt.plot(baseline_fpr, baseline_tpr, lw=2, color='green', label="baseline (AUC:{:.3f})".format(baseline_aucROC))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right",prop={'size': 6})
    filename = directory+'/stacked_roc.png'
    plt.savefig(filename)
    print("saved:{!s}".format(filename))
    plt.figure()
    pred_name = None
    lines = []
    labels = []
    for i in predicates_included:
        for key, value in pred_dic.items():
            if value == i:
                pred_name =key
        l, = plt.plot(fpr[i], tpr[i], lw=2)
        lines.append(l)
        labels.append('ROC for class {} (area = {:.3f})'.format(pred_name, roc_auc[i]))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Predicate ROC")
    plt.legend(lines,labels,loc="upper right",prop={'size': 6})
    filename = directory+'/stacked_predicate_roc.png'
    plt.savefig(filename)
    print("saved:{!s}".format(filename))

ER_MLP_MODEL_HOME='../er_mlp/model/'
PRA_MODEL_HOME='../pra/model/'
er_mlp_features_array_list=[]
for i in args.er_mlp:
    model_base_dir = ER_MLP_MODEL_HOME+i
    fn = open(model_base_dir+'/params.pkl','rb')
    params = pickle.load(fn)
    pred_dic = params['pred_dic']
    pred_index_dic = {}
    for k,v in pred_dic.items():
        pred_index_dic[v] = k

    with open(model_base_dir+'/test/predictions.txt', "r") as _file:
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

# print(e_features)
# print(labels)

print(np.shape(e_features))
pra_features_array_list = []
for i in args.pra:
    model_base_dir = PRA_MODEL_HOME+'/'+i+'/instance/'

    pra_features = []
    for k,v in pred_dic.items():
        with open(model_base_dir+'/test/scores/'+k, "r") as _file:
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

print(np.shape(combined_array))
print(np.shape(y) )

fn = open(args.dir+'/model.pkl','rb')
clf = pickle.load(fn)


y_hat = clf.predict(combined_array)
probabilities = clf.predict_proba( combined_array)[:,1]
# print(y_hat)
print(probabilities)
plot_roc(len(pred_dic), y, probabilities,predicates,pred_dic, args.dir)


    


