import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import interp
import pickle as pickle

def get_roc_stats(params, predicates, predictions, Y):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    predicates_included = []
    for i in range(params['num_preds']):
        predicate_indices = np.where(predicates == i)[0]
        if np.shape(predicate_indices)[0] == 0:
            print('inside')
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
    return fpr["macro"], tpr["macro"], roc_auc["macro"]

plt.rc('legend', fontsize=8) 
ecoliFalse1000 = open('500ecoliFalse1000.txt','rb')
ecoliFalse350_cross = open('5000ecoliFalse350_cross.txt','rb')
ecoliFalse350_over = open('5000ecoliFalse350_over.txt','rb')
ecoliFalse350_weighted = open('5000ecoliFalse350_weighted.txt','rb')
WordnetFalse100 = open('20000WordnetFalse100.txt','rb')
WordnetTrue100 = open('20000WordnetTrue100.txt','rb')
freebaseFalse100 = open('50000freebaseFalse100.txt','rb')
freebaseTrue100 = open('50000freebaseTrue100.txt','rb')

ecoli = pickle.load(ecoliFalse1000)
wordnet= pickle.load(WordnetFalse100)
freebase= pickle.load(freebaseFalse100)
freebaset= pickle.load(freebaseTrue100)
wordnett= pickle.load(WordnetTrue100)

cross= pickle.load(ecoliFalse350_cross)
weight= pickle.load(ecoliFalse350_weighted)
over= pickle.load(ecoliFalse350_over)

e_fpr, e_tpr, e_roc_auc = get_roc_stats(ecoli, ecoli['predicates'], ecoli['prediction_list'], ecoli['labels_test'])
we_fpr, we_tpr, we_roc_auc = get_roc_stats(weight, weight['predicates'], weight['prediction_list'], weight['labels_test'])
c_fpr, c_tpr, c_roc_auc = get_roc_stats(cross, cross['predicates'], cross['prediction_list'], cross['labels_test'])
o_fpr, o_tpr, o_roc_auc = get_roc_stats(over, over['predicates'], over['prediction_list'], over['labels_test'])

plt.figure()
plt.plot(e_fpr, e_tpr, lw=1, label="Ecoli Macro Average ROC curve - Max Margin Loss (AUC:{:.3f})".format(e_roc_auc))
plt.plot(c_fpr, c_tpr, lw=1, label="Ecoli Macro Average ROC curve - Cross Entropy (AUC:{:.3f})".format(c_roc_auc))
plt.plot(we_fpr, we_tpr, lw=1, label="Ecoli Macro Average ROC curve - Weighted Cross Entropy (AUC:{:.3f})".format(we_roc_auc))
plt.plot(o_fpr, o_tpr, lw=1, label="Ecoli Macro Average ROC curve - Cross Entropy with Oversampling (AUC:{:.3f})".format(o_roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ER-MLP ROC")
plt.legend(loc="lower right")
filename = 'er_mlp_roc_ecoli_combined.png'
plt.savefig(filename)
print("saved:{!s}".format(filename))


e_fpr, e_tpr, e_roc_auc = get_roc_stats(ecoli, ecoli['predicates'], ecoli['prediction_list'], ecoli['labels_test'])
w_fpr, w_tpr, w_roc_auc = get_roc_stats(wordnet, wordnet['predicates'], wordnet['prediction_list'], wordnet['labels_test'])
f_fpr, f_tpr, f_roc_auc = get_roc_stats(freebase, freebase['predicates'], freebase['prediction_list'], freebase['labels_test'])
wt_fpr, wt_tpr, wt_roc_auc = get_roc_stats(wordnett, wordnett['predicates'], wordnett['prediction_list'], wordnett['labels_test'])
ft_fpr, ft_tpr, ft_roc_auc = get_roc_stats(freebaset, freebaset['predicates'], freebaset['prediction_list'], freebaset['labels_test'])

plt.figure()
plt.plot(e_fpr, e_tpr, lw=1, label="Ecoli Macro Average ROC curve - ee (AUC:{:.3f})".format(e_roc_auc))
plt.plot(w_fpr, w_tpr, lw=1, label="Wordnet Macro Average ROC curve - ee (AUC:{:.3f})".format(w_roc_auc))
plt.plot(f_fpr, f_tpr, lw=1, label="Freebase Macro Average ROC curve - ee (AUC:{:.3f})".format(f_roc_auc))
plt.plot(wt_fpr, wt_tpr, lw=1, label="Wordnet Macro Average ROC curve - we (AUC:{:.3f})".format(wt_roc_auc))
plt.plot(ft_fpr, ft_tpr, lw=1, label="Freebase Macro Average ROC curve - we (AUC:{:.3f})".format(ft_roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ER-MLP ROC")
plt.legend(loc="lower right")
filename = 'er_mlp_roc_combined.png'
plt.savefig(filename)
print("saved:{!s}".format(filename))



