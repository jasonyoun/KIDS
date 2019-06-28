"""
Filename: evaluate.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Evaluate the performance of the stacked model.

To-do:
    1. Fix variable predicates so that it works in general case.
"""
import os
import sys
import pickle
import argparse
import features
import numpy as np
DIRECTORY = os.path.dirname(__file__)
from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
ABS_PATH_METRICS = os.path.join(DIRECTORY, '../utils')
sys.path.insert(0, ABS_PATH_METRICS)
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats, save_results
from config_parser import ConfigParser

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='build stacked ensemble')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        action='store',
        required=True,
        help='directory to store the model')
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

    model_instance_dir = 'model_instance'
    model_save_dir = os.path.join(model_instance_dir, args.dir)
    config_file = os.path.join(model_save_dir, '{}.ini'.format(args.dir))

    # setup configuration parser
    configparser = ConfigParser(config_file)

    fn = open(os.path.join(model_save_dir, 'model.pkl'), 'rb')
    model_dic = pickle.load(fn)

    pred_dic, test_x, test_y, predicates_test = features.get_x_y(
        configparser.getstr('test_dir'),
        configparser.getstr('er_mlp_model_dir'),
        configparser.getstr('pra_model_dir'),
        args.final_model)

    probabilities = np.zeros_like(predicates_test, dtype=float)
    probabilities = probabilities.reshape((np.shape(probabilities)[0], 1))

    if not args.final_model:
        classifications = np.zeros_like(predicates_test)

    for _, i in pred_dic.items():
        indices, = np.where(predicates_test == i)

        if np.shape(indices)[0] != 0:
            clf = model_dic[i]
            X = test_x[indices]
            predicates = predicates_test[indices]

            probs = clf.predict_proba(X)[:, 1]
            probs = np.reshape(probs, (-1, 1))
            probabilities[indices] = probs

            if not args.final_model:
                classifications[indices] = clf.predict(X)

    if not args.final_model:
        results = {}
        results['predicate'] = {}
        plot_roc(len(pred_dic), test_y, probabilities, predicates, pred_dic, model_save_dir)
        plot_pr(len(pred_dic), test_y, probabilities, predicates, pred_dic, model_save_dir)
        auc_test = roc_auc_stats(len(pred_dic), test_y, probabilities, predicates, pred_dic)
        ap_test = pr_stats(len(pred_dic), test_y, probabilities, predicates, pred_dic)

        fl_measure_test = f1_score(test_y, classifications)
        accuracy_test = accuracy_score(test_y, classifications)
        recall_test = recall_score(test_y, classifications)
        precision_test = precision_score(test_y, classifications)
        confusion_test = confusion_matrix(test_y, classifications)

        results['overall'] = {
            'map': ap_test,
            'roc_auc': auc_test,
            'f1': fl_measure_test,
            'accuracy': accuracy_test,
            'cm': confusion_test,
            'precision': precision_test,
            'recall': recall_test
        }

        for i in range(len(pred_dic)):
            for key, value in pred_dic.items():
                if value == i:
                    pred_name = key

            indices, = np.where(predicates == i)

            if np.shape(indices)[0] != 0:
                classifications_predicate = classifications[indices]
                predicate_predictions = probabilities[indices]
                labels_predicate = test_y[indices]
                fl_measure_predicate = f1_score(labels_predicate, classifications_predicate)
                accuracy_predicate = accuracy_score(labels_predicate, classifications_predicate)
                recall_predicate = recall_score(labels_predicate, classifications_predicate)
                precision_predicate = precision_score(labels_predicate, classifications_predicate)
                confusion_predicate = confusion_matrix(labels_predicate, classifications_predicate)

                print(' - predicate f1 measure for {}: {}'.format(pred_name, fl_measure_predicate))
                print(' - predicate accuracy for {}: {}'.format(pred_name, accuracy_predicate))
                print(' - predicate recall for {}: {}'.format(pred_name, recall_predicate))
                print(' - predicate precision for {}: {}'.format(pred_name, precision_predicate))
                print(' - predicate confusion matrix for {}:'.format(pred_name))
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

    directory = os.path.join(model_save_dir, 'test')
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not args.final_model:
        _file = os.path.join(directory, 'classifications_stacked.txt')
        with open(_file, 'w') as t_f:
            for row in classifications:
                t_f.write(str(row) + '\n')

    _file = os.path.join(directory, 'confidence_stacked.txt')
    with open(_file, 'w') as t_f:
        for row in probabilities:
            t_f.write(str(row) + '\n')

    if not args.final_model:
        save_results(results, directory)

        print('test f1 measure: {}'.format(fl_measure_test))
        print('test accuracy: {}'.format(accuracy_test))
        print('test precision: {}'.format(precision_test))
        print('test recall: {}'.format(recall_test))
        print('test confusion matrix:')
        print('test auc: {}'.format(auc_test))
        print('test ap: {}'.format(ap_test))
        print(confusion_test)
        print(ap_test)

if __name__ == "__main__":
    main()
