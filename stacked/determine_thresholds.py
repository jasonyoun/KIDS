"""
Filename: determine_thresholds.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Find the optimal threshold for the stacked model.

To-do:
"""
import os
import sys
import pickle
import argparse
import features
import numpy as np
DIRECTORY = os.path.dirname(__file__)
from sklearn.metrics import accuracy_score, f1_score
ABS_PATH_METRICS = os.path.join(DIRECTORY, '../utils')
sys.path.insert(0, ABS_PATH_METRICS)
from config_parser import ConfigParser

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='determine thresholds')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        action='store',
        required=True,
        help='directory to store the model')

    return parser.parse_args()

def compute_threshold(predictions_dev, dev_y, f1=True):
    """
    Determine the best threshold to use for classification.

    Inputs:
        predictions_dev: prediction found by running the model
        dev_y: ground truth label to be compared with predictions_list

    Returns:
        best_threshold: threshold that yields the best accuracy
    """
    predictions_list = predictions_dev.reshape(-1, 1)
    dev_labels = dev_y.reshape(-1, 1)
    both = np.column_stack((predictions_list, dev_labels))
    both = both[both[:, 0].argsort()]
    predictions_list = both[:, 0].ravel()
    dev_labels = both[:, 1].ravel()
    accuracies = np.zeros(np.shape(predictions_list))

    for idx in range(np.shape(predictions_list)[0]):
        score = predictions_list[idx]
        predictions = (predictions_list >= score)
        accuracy = accuracy_score(predictions, dev_labels)

        if f1:
            accuracy = f1_score(dev_labels, predictions)

        accuracies[idx] = accuracy

    indices = np.argmax(accuracies)
    best_threshold = np.mean(predictions_list[indices])

    return best_threshold

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

    pred_dic, dev_x, dev_y, predicates_dev = features.get_x_y(
        'dev', configparser.getstr('er_mlp_model_dir'), configparser.getstr('pra_model_dir'))
    # dev_y[:][dev_y[:] == -1] = 0

    predictions_dev = np.zeros_like(predicates_dev, dtype=float)
    predictions_dev = predictions_dev.reshape((np.shape(predictions_dev)[0], 1))
    best_thresholds = np.zeros(len(pred_dic))

    for _, idx in pred_dic.items():
        indices, = np.where(predicates_dev == idx)
        if np.shape(indices)[0] != 0:
            clf = model_dic[idx]
            X = dev_x[indices]
            y = dev_y[indices]

            preds = clf.predict_proba(X)[:, 1]
            preds = preds.reshape((np.shape(preds)[0], 1))
            best_thresholds[idx] = compute_threshold(preds, y, f1=configparser.getbool('F1_FOR_THRESHOLD'))

    print(best_thresholds)

    with open(model_save_dir + '/threshold.pkl', 'wb') as output:
        pickle.dump(best_thresholds, output, pickle.HIGHEST_PROTOCOL)

    print('thresholds saved in: {}'.format(model_save_dir))

if __name__ == "__main__":
    main()
