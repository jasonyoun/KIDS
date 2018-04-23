import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os

from sklearn import utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import random
from scipy import interp
import random
import matplotlib.pyplot as plt


def roc_auc_stats(num_preds, Y, predictions,predicates,pred_dic):
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

    for i in predicates_included:
        for key, value in pred_dic.items():
            if value == i:
                pred_name =key
        print(' - roc auc for class {} (area = {:.3f})'.format(pred_name, roc_auc[i]))

    mean_tpr /= len(predicates_included)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc["macro"]

def pr_stats(num_preds, Y, predictions,predicates,pred_dic):
    baseline = np.zeros(np.shape(predictions))
    baseline_precision, baseline_recall , _ = precision_recall_curve(Y.ravel(), baseline.ravel())
    baseline_aucPR = auc(baseline_recall, baseline_precision)

    precision = dict()
    recall = dict()
    aucPR = dict()
    ap = dict()
    predicates_included = []
    sum_ap = 0.0
    for i in range(num_preds):
        predicate_indices = np.where(predicates == i)[0]
        if np.shape(predicate_indices)[0] == 0:
            continue
        else:
            predicates_included.append(i)
        predicate_predictions = predictions[predicate_indices]
        predicate_labels = Y[predicate_indices]
        precision[i], recall[i] , _ = precision_recall_curve(predicate_labels.ravel(), predicate_predictions.ravel())
        ap[i] = average_precision_score(predicate_labels.ravel(), predicate_predictions.ravel())
        sum_ap+=ap[i]
        aucPR[i] = auc(recall[i], precision[i])

    for i in predicates_included:
        for key, value in pred_dic.items():
            if value == i:
                pred_name =key
        print(' - PR auc for class {} (area = {:.3f})'.format(pred_name, ap[i]))

    mean_average_precision = sum_ap/len(predicates_included)
    return mean_average_precision




def plot_roc(num_preds, Y, predictions,predicates,pred_dic, directory):
    baseline = np.zeros(np.shape(predictions))
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
    filename = directory+'/er_mlp_roc.png'
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
    filename = directory+'/er_mlp_predicate_roc.png'
    plt.savefig(filename)
    print("saved:{!s}".format(filename))

def plot_pr(num_preds, Y, predictions,predicates,pred_dic, directory):
    baseline = np.zeros(np.shape(predictions))
    baseline_precision, baseline_recall , _ = precision_recall_curve(Y.ravel(), baseline.ravel())
    baseline_aucPR = auc(baseline_recall, baseline_precision)

    precision = dict()
    recall = dict()
    aucPR = dict()
    predicates_included = []
    sum_ap = 0.0
    ap = dict()
    for i in range(num_preds):
        predicate_indices = np.where(predicates == i)[0]
        if np.shape(predicate_indices)[0] == 0:
            print('inside')
            continue
        else:
            predicates_included.append(i)
        predicate_predictions = predictions[predicate_indices]
        predicate_labels = Y[predicate_indices]
        precision[i], recall[i] , _ = precision_recall_curve(predicate_labels.ravel(), predicate_predictions.ravel())
        aucPR[i] = auc(recall[i], precision[i])
        ap[i] = average_precision_score(predicate_labels.ravel(), predicate_predictions.ravel())
        sum_ap+=ap[i]
    mean_average_precision = sum_ap/len(predicates_included)
    plt.figure()
    pred_name = None
    lines = []
    labels = []
    for i in predicates_included:
        for key, value in pred_dic.items():
            if value == i:
                pred_name =key
        l, = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {} (area = {:.3f})'.format(pred_name, aucPR[i]))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall (mAP = {:.3f})".format( mean_average_precision))
    plt.legend(lines,labels,loc="upper right",prop={'size': 6})
    filename = directory+'/er_mlp_pr.png'
    plt.savefig(filename)
    print("saved:{!s}".format(filename))

def plot_cost( iterations, cost_list, directory):
    plt.figure()
    plt.plot(iterations, cost_list, lw=1, color='darkorange')
    plt.xlabel("Iteration #")
    plt.ylabel("Loss")
    plt.title("Loss per Iteration of Training")
    filename = directory+'/cost_.png'
    plt.savefig(filename)



