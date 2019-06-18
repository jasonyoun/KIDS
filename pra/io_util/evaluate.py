"""
Filename: evaluate.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Perform evaluation.

To-do:
"""
import os
import sys
import argparse
import numpy as np
DIRECTORY = os.path.dirname(__file__)
ABS_PATH_METRICS = os.path.join(DIRECTORY, '../../utils')
sys.path.insert(0, ABS_PATH_METRICS)
from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats, save_results

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='evaluate the results')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        default='./',
        help='base directory')

    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_argument()

    with open('./selected_relations') as _file:
        relations = _file.readlines()

    relations = [x.strip() for x in relations]

    index = 0
    predicates_dic = {}
    for relation in relations:
        predicates_dic[relation] = index
        index += 1

    combined_scores_array = None
    combined_predicates_array = None
    combined_labels_array = None
    combined_classifications_array = None
    start = 0

    for k, v in predicates_dic.items():
        with open(args.dir + '/scores/' + k, "r") as _file, open(args.dir + '/queriesR_labels/' + k, "r") as l_file, open(args.dir + '/classifications/' + k, "r") as c_file:
            scores = _file.readlines()
            scores = [x.strip().split('\t')[0] for x in scores]

            labels = l_file.readlines()
            labels = [x.strip().split('\t')[2] for x in labels]

            classifications = c_file.readlines()
            classifications = [x.strip().split('\t')[0] for x in classifications]

            predicates = [v for x in scores]
            predicates_array = np.array(predicates)
            scores_array = np.array(scores)
            labels_array = np.array(labels)
            classifications_array = np.array(classifications)

            if start == 0:
                combined_scores_array = scores_array
                combined_predicates_array = predicates_array
                combined_labels_array = labels_array
                combined_classifications_array = classifications_array
                start += 1
            else:
                combined_scores_array = np.append(combined_scores_array, scores_array)
                combined_predicates_array = np.append(combined_predicates_array, predicates_array)
                combined_labels_array = np.append(combined_labels_array, labels_array)
                combined_classifications_array = np.append(combined_classifications_array, classifications_array)

    combined_scores_array = np.transpose(combined_scores_array).astype(float)
    combined_predicates_array = np.transpose(combined_predicates_array).astype(int)
    combined_labels_array = np.transpose(combined_labels_array).astype(int)
    combined_classifications_array = np.transpose(combined_classifications_array).astype(int)
    combined_labels_array[:][combined_labels_array[:] == -1] = 0

    results = {}
    results['predicate'] = {}

    for i in range(len(predicates_dic)):
        for key, value in predicates_dic.items():
            if value == i:
                pred_name = key

        indices, = np.where(combined_predicates_array == i)
        classifications_predicate = combined_classifications_array[indices]
        labels_predicate = combined_labels_array[indices]
        predicate_predictions = combined_scores_array[indices]
        classifications_predicate[:][classifications_predicate[:] == -1] = 0

        fl_measure_predicate = f1_score(labels_predicate, classifications_predicate)
        accuracy_predicate = accuracy_score(labels_predicate, classifications_predicate)
        recall_predicate = recall_score(labels_predicate, classifications_predicate)
        precision_predicate = precision_score(labels_predicate, classifications_predicate)
        confusion_predicate = confusion_matrix(labels_predicate, classifications_predicate)

        print(' - test f1 measure for {}: {}'.format(pred_name, fl_measure_predicate))
        print(' - test accuracy for {}: {}'.format(pred_name, accuracy_predicate))
        print(' - test precision for {}: {}'.format(pred_name, precision_predicate))
        print(' - test recall for {}: {}'.format(pred_name, recall_predicate))
        print(' - test confusion matrix for {}:'.format(pred_name))
        print(confusion_predicate)
        print(' ')

        fpr_pred, tpr_pred, _ = roc_curve(labels_predicate.ravel(), predicate_predictions.ravel())
        roc_auc_pred = auc(fpr_pred, tpr_pred)
        ap_pred = average_precision_score(labels_predicate.ravel(), predicate_predictions.ravel())

        results['predicate'][pred_name] = {
            'map': ap_pred,
            'roc_auc': roc_auc_pred,
            'f1': fl_measure_predicate,
            'accuracy': accuracy_predicate,
            'cm': confusion_predicate,
            'precision': precision_predicate,
            'recall': recall_predicate
        }

    mean_average_precision_test = pr_stats(len(relations), combined_labels_array, combined_scores_array, combined_predicates_array, predicates_dic)
    roc_auc_test = roc_auc_stats(len(relations), combined_labels_array, combined_scores_array, combined_predicates_array, predicates_dic)
    fl_measure_test = f1_score(combined_labels_array, combined_classifications_array)
    accuracy_test = accuracy_score(combined_labels_array, combined_classifications_array)
    recall_test = recall_score(combined_labels_array, combined_classifications_array)
    precision_test = precision_score(combined_labels_array, combined_classifications_array)
    confusion_test = confusion_matrix(combined_labels_array, combined_classifications_array)
    plot_pr(len(relations), combined_labels_array, combined_scores_array, combined_predicates_array, predicates_dic, args.dir, name_of_file='pra_not_calibrated')
    plot_roc(len(relations), combined_labels_array, combined_scores_array, combined_predicates_array, predicates_dic, args.dir, name_of_file='pra_not_calibrated')

    results['overall'] = {
        'map': mean_average_precision_test,
        'roc_auc': roc_auc_test,
        'f1': fl_measure_test,
        'accuracy': accuracy_test,
        'cm': confusion_test,
        'precision': precision_test,
        'recall': recall_test
    }

    print('test mean average precision: {}'.format(mean_average_precision_test))
    print('test f1 measure: {}'.format(fl_measure_test))
    print('test accuracy: {}'.format(accuracy_test))
    print('test roc auc: {}'.format(roc_auc_test))
    print('test precision: {}'.format(precision_test))
    print('test recall: {}'.format(recall_test))
    print('test confusion matrix:')
    print(confusion_test)
    print(' ')
    save_results(results, args.dir)

    _file = args.dir + "/classifications_pra.txt"
    with open(_file, 'w') as t_f:
        for row in classifications:
            t_f.write(str(row) + '\n')

if __name__ == "__main__":
    main()
