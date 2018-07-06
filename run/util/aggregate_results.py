import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
import configparser
import argparse
import copy
import glob
from scipy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

directory = os.path.dirname(__file__)
abs_path_metrics= os.path.join(directory, '../../utils')
sys.path.insert(0, abs_path_metrics)
from metrics import  save_results

config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description='evaluate')
parser.add_argument('--dir', metavar='dir', nargs='+', default='./',
                    help='base directory')
parser.add_argument('--results_dir', metavar='dir', nargs='?', action='store',required=True,
                    help='directory to store the model')

args = parser.parse_args()
results_dir = args.results_dir

def load_results(_file):
    with open(_file, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def get_latest_fig(path, prefix):
    list_of_files = glob.glob(path+'/'+prefix+'*.pkl')
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'rb') as pickle_file:
        return pickle.load(pickle_file)
def add_result(results,results_overall):
    for metric,value in results['overall'].items():
        results_overall['overall'][metric].append(value)
    for predicate,metrics in results['predicate'].items():
        for metric,value in metrics.items():
            results_overall['predicate'][predicate][metric].append(value)

def create_results_list(results):
    results_list = {}
    results_list['overall'] = {}
    for metric,value in results['overall'].items():
        results_list['overall'][metric]=[]
    results_list['predicate'] = {}
    for predicate,metrics in results['predicate'].items():
        results_list['predicate'][predicate] = {}
        for metric,value in metrics.items():
            results_list['predicate'][predicate][metric]=[]
    return results_list


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate(results_list,size):
    results_overall= {}
    results_overall['overall'] = {}
    for metric,value in results_list['overall'].items():
        if metric=='map' or metric=='roc_auc':
            results_overall['overall'][metric] = np.mean(np.array(results_list['overall'][metric]))
            results_overall['overall']['{}_std'.format(metric)] = np.std(np.array(results_list['overall'][metric]))
        elif metric=='cm':
            precision = [v[1][1]/(v[1][1] +v[0][1]) for v in value]
            recall = [v[1][1]/(v[1][1] +v[1][0]) for v in value]
            accuracy = [(v[1][1]+v[0][0])/(v[1][1] +v[1][0]+v[0][1]+v[0][0]) for v in value]
            known_true = [v[1][0]+v[1][1] for v in value]
            known_neg = [v[0][0]+v[0][1] for v in value]
            predicted_true = [v[0][1]+v[1][1] for v in value]
            predicted_neg = [v[0][0]+v[1][0] for v in value]
            specificity = [v[0][0]/(v[0][1] +v[0][0]) for v in value]
            npv = [v[0][0]/(v[0][0] +v[1][0]) for v in value]
            fdr = [t/n for t,n in zip(predicted_true, predicted_neg)]
            results_overall['overall']['cm'] = np.mean(np.array(value),axis=0)
            results_overall['overall']['cm_std'] = np.std(np.array(value),axis=0)
            results_overall['overall']['precision'] = np.mean(np.array(precision))
            results_overall['overall']['precision_std'] = np.std(np.array(precision))
            results_overall['overall']['recall'] = np.mean(np.array(recall))
            results_overall['overall']['recall_std'] = np.std(np.array(recall))
            results_overall['overall']['accuracy'] = np.mean(np.array(accuracy))
            results_overall['overall']['accuracy_std'] = np.std(np.array(accuracy))

            results_overall['overall']['known_true'] = np.mean(np.array(known_true))
            results_overall['overall']['known_true_std'] = np.std(np.array(known_true))
            results_overall['overall']['known_neg'] = np.mean(np.array(known_neg))
            results_overall['overall']['known_neg_std'] = np.std(np.array(known_neg))
            results_overall['overall']['predicted_true'] = np.mean(np.array(predicted_true))
            results_overall['overall']['predicted_true_std'] = np.std(np.array(predicted_true))
            results_overall['overall']['predicted_neg'] = np.mean(np.array(predicted_neg))
            results_overall['overall']['predicted_neg_std'] = np.std(np.array(predicted_neg))
            results_overall['overall']['specificity'] = np.mean(np.array(specificity))
            results_overall['overall']['specificity_std'] = np.std(np.array(specificity))
            results_overall['overall']['npv'] = np.mean(np.array(npv))
            results_overall['overall']['npv_std'] = np.std(np.array(npv))
            results_overall['overall']['fdr'] = np.mean(np.array(fdr))
            results_overall['overall']['fdr_std'] = np.std(np.array(fdr))

            sum_p_r = [p+r for p,r in zip(precision, recall)]
            product_p_r = [2*p*r for p,r in zip(precision, recall)]
            f1 = [p/s for p,s in zip(product_p_r, sum_p_r)]
            results_overall['overall']['f1'] = np.mean(np.array(f1))
            results_overall['overall']['f1_std'] = np.std(np.array(f1))
    results_overall['predicate'] = {}
    for predicate,metrics in results_list['predicate'].items():
        results_overall['predicate'][predicate] = {}
        for metric,value in metrics.items():
            if metric=='map' or metric=='roc_auc':
                results_overall['predicate'][predicate][metric] = np.mean(np.array(results_list['overall'][metric]))
                results_overall['predicate'][predicate]['{}_std'.format(metric)] = np.std(np.array(results_list['overall'][metric]))
            elif metric=='cm':
                precision = [v[1][1]/(v[1][1] +v[0][1]) for v in value]
                recall = [v[1][1]/(v[1][1] +v[1][0]) for v in value]
                accuracy = [(v[1][1]+v[0][0])/(v[1][1] +v[1][0]+v[0][1]+v[0][0]) for v in value]
                known_true = [v[1][0]+v[1][1] for v in value]
                known_neg = [v[0][0]+v[0][1] for v in value]
                predicted_true = [v[0][1]+v[1][1] for v in value]
                predicted_neg = [v[0][0]+v[1][0] for v in value]
                specificity = [v[0][0]/(v[0][1] +v[0][0]) for v in value]
                npv = [v[0][0]/(v[0][0] +v[1][0]) for v in value]
                fdr = [t/n for t,n in zip(predicted_true, predicted_neg)]
                results_overall['predicate'][predicate]['cm'] = np.mean(np.array(value),axis=0)
                results_overall['predicate'][predicate]['cm_std'] = np.std(np.array(value),axis=0)
                results_overall['predicate'][predicate]['precision'] = np.mean(np.array(precision))
                results_overall['predicate'][predicate]['precision_std'] = np.std(np.array(precision))
                results_overall['predicate'][predicate]['recall'] = np.mean(np.array(recall))
                results_overall['predicate'][predicate]['recall_std'] = np.std(np.array(recall))
                results_overall['predicate'][predicate]['accuracy'] = np.mean(np.array(accuracy))
                results_overall['predicate'][predicate]['accuracy_std'] = np.std(np.array(accuracy))

                results_overall['predicate'][predicate]['known_true'] = np.mean(np.array(known_true))
                results_overall['predicate'][predicate]['known_true_std'] = np.std(np.array(known_true))
                results_overall['predicate'][predicate]['known_neg'] = np.mean(np.array(known_neg))
                results_overall['predicate'][predicate]['known_neg_std'] = np.std(np.array(known_neg))
                results_overall['predicate'][predicate]['predicted_true'] = np.mean(np.array(predicted_true))
                results_overall['predicate'][predicate]['predicted_true_std'] = np.std(np.array(predicted_true))
                results_overall['predicate'][predicate]['predicted_neg'] = np.mean(np.array(predicted_neg))
                results_overall['predicate'][predicate]['predicted_neg_std'] = np.std(np.array(predicted_neg))
                results_overall['predicate'][predicate]['specificity'] = np.mean(np.array(specificity))
                results_overall['predicate'][predicate]['specificity_std'] = np.std(np.array(specificity))
                results_overall['predicate'][predicate]['npv'] = np.mean(np.array(npv))
                results_overall['predicate'][predicate]['npv_std'] = np.std(np.array(npv))
                results_overall['predicate'][predicate]['fdr'] = np.mean(np.array(fdr))
                results_overall['predicate'][predicate]['fdr_std'] = np.std(np.array(fdr))

                sum_p_r = [p+r for p,r in zip(precision, recall)]
                product_p_r = [2*p*r for p,r in zip(precision, recall)]
                f1 = [p/s for p,s in zip(product_p_r, sum_p_r)]
                results_overall['predicate'][predicate]['f1'] = np.mean(np.array(f1))
                results_overall['predicate'][predicate]['f1_std'] = np.std(np.array(f1))
    return results_overall

first = 1
root_dir= os.path.join(directory, '../..')
for fold in args.dir:
    pra_results = load_results(root_dir+'/pra/model/model_instance/'+fold+'/instance/test/results/results.pkl')
    er_mlp_results = load_results(root_dir+'/er_mlp/model/model_instance/'+fold+'/test/results/results.pkl')
    stacked_results = load_results(root_dir+'/stacked/model_instance/'+fold+'/test/results/results.pkl')
    if first==1:
        pra_results_list = create_results_list(pra_results)
        er_mlp_results_list=  create_results_list(er_mlp_results)
        stacked_results_list = create_results_list(stacked_results)
        first+=1
    else:
        add_result(pra_results,pra_results_list)
        add_result(er_mlp_results,er_mlp_results_list)
        add_result(stacked_results,stacked_results_list)
create_dir(results_dir)
pra_dir =results_dir+'/pra'
er_mlp_dir =results_dir+'/er_mlp'
stacked_dir =results_dir+'/stacked'
create_dir(pra_dir)
create_dir(er_mlp_dir)
create_dir(stacked_dir)
pra_results_overall = calculate(pra_results_list, len(args.dir))
er_mlp_results_overall = calculate(er_mlp_results_list, len(args.dir))
stacked_results_overall = calculate(stacked_results_list, len(args.dir))
save_results(pra_results_overall,pra_dir)
save_results(er_mlp_results_overall,er_mlp_dir)
save_results(stacked_results_overall,stacked_dir)

def get_mean_average_roc(roc):
    all_fpr = {}
    mean_tpr = {}
    predicates_fpr = {}
    predicates_tpr = {}
    predicates_auc = {}
    mean_fpr = {}
    mean_tpr = {}
    mean_auc = {}
    mean_auc_std ={}
    for fold in args.dir:
        for predicate, values in roc[fold].items():
            fpr, tpr, _auc = values
            if predicate not in predicates_fpr:
                predicates_auc[predicate] = []
                predicates_fpr[predicate] = []
                predicates_tpr[predicate] = []
            predicates_auc[predicate].append(_auc)
            predicates_fpr[predicate].append(fpr)
            predicates_tpr[predicate].append(tpr)
        for predicate,fpr_list in predicates_fpr.items():
            all_fpr[predicate] = np.unique(np.concatenate(fpr_list))
            mean_tpr[predicate] = np.zeros_like(all_fpr[predicate])

    for predicate,tprs in predicates_tpr.items():
        for i in range(len(tprs)):
            mean_tpr[predicate] += interp(all_fpr[predicate], predicates_fpr[predicate][i], tprs[i])
        mean_tpr[predicate] /= len(tprs)
        mean_fpr[predicate] = all_fpr[predicate]
        mean_tpr[predicate] = mean_tpr[predicate]
        mean_auc[predicate] = auc(mean_fpr[predicate], mean_tpr[predicate])
        mean_auc_std[predicate] = np.std(np.array(predicates_auc[predicate]))
    return mean_fpr, mean_tpr, mean_auc, mean_auc_std

# Can't simply linearly interpolate the pr curve. Plotting all curves
# def get_mean_average_pr(pr):
#     all_recall = {}
#     mean_precision = {}
#     predicates_recall = {}
#     predicates_precision = {}
#     predicates_ap = {}
#     mean_recall = {}
#     mean_precision = {}
#     mean_ap = {}
#     mean_ap_std ={}
#     for fold in args.dir:
#         for predicate, values in pr[fold].items():
#             recall, precision,ap = values
#             if predicate not in predicates_recall:
#                 predicates_ap[predicate] = []
#                 predicates_recall[predicate] = []
#                 predicates_precision[predicate] = []
#             predicates_ap[predicate].append(ap)
#             predicates_recall[predicate].append(recall)
#             predicates_precision[predicate].append(precision)
#         for predicate,recall_list in predicates_recall.items():
#             all_recall[predicate] = np.unique(np.concatenate(recall_list))
#             mean_precision[predicate] = np.zeros_like(all_recall[predicate])

#     for predicate,precisions in predicates_precision.items():
#         for i in range(len(precisions)):
#             mean_precision[predicate] += interp(all_recall[predicate], predicates_recall[predicate][i], precisions[i])
#         mean_precision[predicate] /= len(precisions)
#         mean_recall[predicate] = all_recall[predicate]
#         mean_precision[predicate] = mean_precision[predicate]
#         mean_ap[predicate] = np.mean(np.array(predicates_ap[predicate]))
#         mean_ap_std[predicate] = np.std(np.array(predicates_ap[predicate]))
#     return mean_recall, mean_precision, mean_ap, mean_ap_std

def plot_pr(pr,model,results_dir):
    plt.figure()
    for fold in args.dir:
        # There only exists one predicate: Confers resistance to antibiotic. No need to show the name in the plot
        for predicate, values in pr[fold].items():
            recall, precision,ap = values
            plt.step(recall,precision, label="{} (AP:{:.3f})".format(fold,ap),where='post')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("{} - Precision Recall".format(model))
            plt.legend(loc="upper right",prop={'size': 6})
            filename = results_dir+'/{}_pr.png'.format(model)
            plt.savefig(filename)

pra_pr,pra_roc,er_mlp_pr,er_mlp_roc,stacked_pr,stacked_roc = {},{},{},{},{},{}
for fold in args.dir:
    pra_pr[fold] = get_latest_fig(root_dir+'/pra/model/model_instance/'+fold+'/instance/test/fig', 'pr_pra')
    pra_roc[fold] = get_latest_fig(root_dir+'/pra/model/model_instance/'+fold+'/instance/test/fig', 'roc_pra')
    er_mlp_pr[fold] = get_latest_fig(root_dir+'/er_mlp/model/model_instance/'+fold+'/test/fig', 'pr_er_mlp')
    er_mlp_roc[fold] = get_latest_fig(root_dir+'/er_mlp/model/model_instance/'+fold+'/test/fig', 'roc_er_mlp')
    stacked_pr[fold] = get_latest_fig(root_dir+'/stacked/model_instance/'+fold+'/fig', 'pr_model')
    stacked_roc[fold] = get_latest_fig(root_dir+'/stacked/model_instance/'+fold+'/fig', 'roc_model')

pra_mean_fpr, pra_mean_tpr, pra_mean_auc, pra_mean_auc_std = get_mean_average_roc(pra_roc)
plot_pr(pra_pr, 'PRA',results_dir)

er_mlp_mean_fpr, er_mlp_mean_tpr, er_mlp_mean_auc, er_mlp_mean_auc_std = get_mean_average_roc(er_mlp_roc)
plot_pr(er_mlp_pr, 'ER-MLP',results_dir)

stacked_mean_fpr, stacked_mean_tpr, stacked_mean_auc, stacked_mean_auc_std = get_mean_average_roc(stacked_roc)
plot_pr(stacked_pr, 'Stacked',results_dir)


# plt.figure()
# for predicate,predicate_precision in pra_mean_precision.items():
#     plt.plot(pra_mean_recall[predicate],pra_mean_precision[predicate], lw=2, label="PRA (AP:{:.3f} +- {:.3f})".format(pra_mean_ap[predicate],pra_mean_ap_std[predicate]))
#     plt.plot(er_mlp_mean_recall[predicate],er_mlp_mean_precision[predicate], lw=2, label="ER-MLP (AP:{:.3f} +- {:.3f})".format(er_mlp_mean_ap[predicate],er_mlp_mean_ap_std[predicate]))
#     plt.plot(stacked_mean_recall[predicate],stacked_mean_precision[predicate], lw=2,label="Stacked (AP:{:.3f} +- {:.3f})".format(stacked_mean_ap[predicate],stacked_mean_ap_std[predicate]))
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision Recall")
# plt.legend(loc="upper right",prop={'size': 6})
# filename = results_dir+'/pr.png'
# plt.savefig(filename)


plt.figure()
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="baseline (AUC:{:.3f})".format(0.5))
for predicate,predicate_tpr in pra_mean_tpr.items():
    plt.plot(pra_mean_fpr[predicate],pra_mean_tpr[predicate], label="PRA (AUC:{:.3f} +- {:.3f})".format(pra_mean_auc[predicate],pra_mean_auc_std[predicate]))
    plt.plot(er_mlp_mean_fpr[predicate],er_mlp_mean_tpr[predicate], label="ER-MLP (AUC:{:.3f} +- {:.3f})".format(er_mlp_mean_auc[predicate],er_mlp_mean_auc_std[predicate]))
    plt.plot(stacked_mean_fpr[predicate],stacked_mean_tpr[predicate], label="Stacked (AUC:{:.3f} +- {:.3f})".format(stacked_mean_auc[predicate],stacked_mean_auc_std[predicate]))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right",prop={'size': 6})
filename = results_dir+'/roc.png'
plt.savefig(filename)





