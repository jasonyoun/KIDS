import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import interp

plt.rc('legend', fontsize=8)  

class Plotter:
    def __init__(self,params):
        self.params = params


    # plot the roc
    def plot_roc( self,Y, predictions,predicates):
        baseline = np.zeros(np.shape(predictions))
        baseline_fpr, baseline_tpr , _ = roc_curve(Y.ravel(), baseline.ravel())
        baseline_aucROC = auc(baseline_fpr, baseline_tpr)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        predicates_included = []
        for i in range(self.params['num_preds']):
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
        plt.figure()
        plt.plot(fpr["macro"], tpr["macro"], lw=2, color='darkorange', label="Macro Average ROC curve (AUC:{:.3f})".format(roc_auc["macro"]))
        plt.plot(baseline_fpr, baseline_tpr, lw=2, color='green', label="baseline (AUC:{:.3f})".format(baseline_aucROC))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{} ER-MLP ROC (epochs:{:d}, batch size:{:d})".format(self.params['data_type'],self.params['training_epochs'],self.params['batch_size']))
        plt.legend(loc="lower right")
        filename = 'fig/er_mlp_roc_{}.png'.format(self.params['data_type'])
        plt.savefig(filename)
        print("saved:{!s}".format(filename))

    # plot the precision recall curve
    def plot_pr( self, Y, predictions,predicates):
        baseline = np.zeros(np.shape(predictions))
        baseline_precision, baseline_recall , _ = precision_recall_curve(Y.ravel(), baseline.ravel())
        baseline_aucPR = auc(baseline_recall, baseline_precision)

        precision = dict()
        recall = dict()
        aucPR = dict()
        predicates_included = []
        for i in range(self.params['num_preds']):
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
        plt.figure()
        pred_name = None
        lines = []
        labels = []
        for i in predicates_included:
            for key, value in self.params['pred_dic'].items():
                if value == i:
                    pred_name =key
            l, = plt.plot(recall[i], precision[i], lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {} (area = {:.3f})'.format(pred_name, aucPR[i]))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("{} - ER-MLP PR  (epochs:{:d}, batch size:{:d}) ".format(self.params['data_type'],self.params['training_epochs'],self.params['batch_size']))
        plt.legend(lines,labels,loc="upper right")
        filename = 'fig/er_mlp_pr_{}.png'.format(self.params['data_type'])
        plt.savefig(filename)
        print("saved:{!s}".format(filename))

    # plot the cost per iteration
    def plot_cost( self, iterations, cost_list):
        plt.figure()
        plt.plot(iterations, cost_list, lw=1, color='darkorange')
        plt.xlabel("Iteration #")
        plt.ylabel("Loss")
        plt.title("Loss per Iteration of Training")
        filename = 'fig/cost_{}.png'.format(self.params['data_type'])
        plt.savefig(filename)
