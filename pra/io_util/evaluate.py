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
    parser.add_argument(
        '--final_model',
        default=False,
        action='store_true',
        help='Set when training the final model')

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
        _file = open(args.dir + '/scores/' + k, "r")
        l_file = open(args.dir + '/queriesR_labels/' + k, "r")
        if not args.final_model:
            c_file = open(args.dir + '/classifications/' + k, "r")

        scores = _file.readlines()
        _file.close()
        scores = [x.strip().split('\t')[0] for x in scores]

        labels = l_file.readlines()
        l_file.close()
        labels = [x.strip().split('\t')[2] for x in labels]

        if not args.final_model:
            classifications = c_file.readlines()
            c_file.close()
            classifications = [x.strip().split('\t')[0] for x in classifications]

        predicates = [v for x in scores]
        predicates_array = np.array(predicates)
        scores_array = np.array(scores)
        labels_array = np.array(labels)
        if not args.final_model:
            classifications_array = np.array(classifications)

        if start == 0:
            combined_scores_array = scores_array
            combined_predicates_array = predicates_array
            combined_labels_array = labels_array
            if not args.final_model:
                combined_classifications_array = classifications_array
            start += 1
        else:
            combined_scores_array = np.append(combined_scores_array, scores_array)
            combined_predicates_array = np.append(combined_predicates_array, predicates_array)
            combined_labels_array = np.append(combined_labels_array, labels_array)
            if not args.final_model:
                combined_classifications_array = np.append(combined_classifications_array, classifications_array)

    combined_scores_array = np.transpose(combined_scores_array).astype(float)
    combined_predicates_array = np.transpose(combined_predicates_array).astype(int)
    combined_labels_array = np.transpose(combined_labels_array).astype(int)
    combined_labels_array[:][combined_labels_array[:] == -1] = 0
    if not args.final_model:
        combined_classifications_array = np.transpose(combined_classifications_array).astype(int)

    results = {}
    results['predicate'] = {}

    for i in range(len(predicates_dic)):
        for key, value in predicates_dic.items():
            if value == i:
                pred_name = key

        indices, = np.where(combined_predicates_array == i)
        labels_predicate = combined_labels_array[indices]
        predicate_predictions = combined_scores_array[indices]
        if not args.final_model:
            classifications_predicate = combined_classifications_array[indices]
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
        }

        if not args.final_model:
            results['predicate'][pred_name]['f1'] = fl_measure_predicate,
            results['predicate'][pred_name]['accuracy'] = accuracy_predicate,
            results['predicate'][pred_name]['cm'] = confusion_predicate,
            results['predicate'][pred_name]['precision'] = precision_predicate,
            results['predicate'][pred_name]['recall'] = recall_predicate

    mean_average_precision_test = pr_stats(len(relations), combined_labels_array, combined_scores_array, combined_predicates_array, predicates_dic)
    roc_auc_test = roc_auc_stats(len(relations), combined_labels_array, combined_scores_array, combined_predicates_array, predicates_dic)
    if not args.final_model:
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
    }

    if not args.final_model:
        results['overall']['f1'] = fl_measure_test,
        results['overall']['accuracy'] = accuracy_test,
        results['overall']['cm'] = confusion_test,
        results['overall']['precision'] = precision_test,
        results['overall']['recall'] = recall_test

    print('test mean average precision: {}'.format(mean_average_precision_test))
    print('test roc auc: {}'.format(roc_auc_test))
    if not args.final_model:
        print('test f1 measure: {}'.format(fl_measure_test))
        print('test accuracy: {}'.format(accuracy_test))
        print('test precision: {}'.format(precision_test))
        print('test recall: {}'.format(recall_test))
        print('test confusion matrix:')
        print(confusion_test)
    print(' ')

    save_results(results, args.dir)

    if not args.final_model:
        with open(args.dir + "/classifications_pra.txt", 'w') as t_f:
            for row in classifications:
                t_f.write(str(row) + '\n')

if __name__ == "__main__":
    main()
