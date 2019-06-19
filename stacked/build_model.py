"""
Filename: build_model.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Build stacked ensemble using AdaBoost.

To-do:
"""
import os
import sys
import pickle
import argparse
import numpy as np
DIRECTORY = os.path.dirname(__file__)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
sys.path.insert(0, os.path.join(DIRECTORY, '../utils'))
import features
from imblearn.over_sampling import SMOTE
from config_parser import ConfigParser
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='build stacked ensemble')

    parser.add_argument(
        '--pra',
        metavar='pra_model (pra_model_2)',
        nargs='?',
        action='store',
        required=True,
        help='The pra models to add')

    parser.add_argument(
        '--er_mlp',
        metavar='er_mlp_model (er_mlp_model_2)',
        nargs='?',
        action='store',
        required=True,
        help='The er-mlp models to add')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        action='store',
        required=True,
        help='directory to store the model')

    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_argument()

    # paths
    model_instance_dir = 'model_instance'
    model_save_dir = os.path.join(model_instance_dir, args.dir)
    config_file = os.path.join(model_save_dir, '{}.ini'.format(args.dir))

    # setup configuration parser
    configparser = ConfigParser(config_file)

    def perform_randomized_search(train_x, train_y, test_x, test_y):
        """
        Given the train & test set, perform randomized search to find
        best n_estimators & learning_rate for the adaboost algorithm.

        Inputs:
            train_x: numpy array where
                train_x[:, 0] = er_mlp prediction raw output
                train_x[:, 1] = pra prediction raw output
                train_x[:, 2] = valid / invalid depending on pra
            train_y: numpy array containing the ground truth label
            test_x: same as train_x but for test data
            test_y: same as train_y but for test data

        Returns:
            dictionary of numpy (masked) ndarrays
            containing the search results
        """

        def get_results_of_search(results, count=5):
            # roc_auc
            print("")
            print("")
            print("roc_auc")
            print("")
            for idx in range(1, count + 1):
                runs = np.flatnonzero(results['rank_test_roc_auc'] == idx)
                for run in runs:
                    print("evaluation rank: {}".format(idx))
                    print("score: {}".format(results['mean_test_roc_auc'][run]))
                    print("std: {}".format(results['std_test_roc_auc'][run]))
                    print(results['params'][run])
                    print("")

            # f1
            print("")
            print("")
            print("f1")
            print("")
            for idx in range(1, count + 1):
                runs = np.flatnonzero(results['rank_test_f1'] == idx)
                for run in runs:
                    print("evaluation rank: {}".format(idx))
                    print("score: {}".format(results['mean_test_f1'][run]))
                    print("std: {}".format(results['std_test_f1'][run]))
                    print(results['params'][run])
                    print("")

                    # # use F1 for reporting the results
                    # if idx==1:
                    #   ap_params = results['params'][run]

            # average precision
            print("")
            print("")
            print("average precision")
            print("")
            for idx in range(1, count + 1):
                runs = np.flatnonzero(results['rank_test_average_precision'] == idx)
                for run in runs:
                    print("evaluation rank: {}".format(idx))
                    print("score: {}".format(results['mean_test_average_precision'][run]))
                    print("std: {}".format(results['std_test_average_precision'][run]))
                    print(results['params'][run])
                    print("")

                    # use average precision for reporting the results
                    if idx == 1:
                        ap_params = results['params'][run]

            # accuracy
            print("")
            print("")
            print("accuracy")
            print("")
            for idx in range(1, count + 1):
                runs = np.flatnonzero(results['rank_test_accuracy'] == idx)
                for run in runs:
                    print("evaluation rank: {}".format(idx))
                    print("score: {}".format(results['mean_test_accuracy'][run]))
                    print("std: {}".format(results['std_test_accuracy'][run]))
                    print(results['params'][run])
                    print("")

            return ap_params

        all_x = np.vstack((train_x, test_x))
        all_y = np.vstack((train_y, test_y))
        all_y = all_y.astype(int)

        # get train / test split indices for predefined split cross-validator
        train_indices = np.full(np.shape(train_x)[0], -1)
        test_indices = np.full(np.shape(test_x)[0], 0)
        test_fold = np.hstack((train_indices, test_indices))

        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=0)

        param_distribution = {
            'learning_rate': np.arange(
                float(configparser.getfloat('START', 'RANDOM_SEARCH_LEARNING_RATE')),
                float(configparser.getfloat('END', 'RANDOM_SEARCH_LEARNING_RATE')),
                float(configparser.getfloat('INCREMENT', 'RANDOM_SEARCH_LEARNING_RATE'))),

            'n_estimators': np.arange(
                int(configparser.getint('START', section='RANDOM_SEARCH_ESTIMATORS')),
                int(configparser.getint('END', section='RANDOM_SEARCH_ESTIMATORS')),
                int(configparser.getint('INCREMENT', section='RANDOM_SEARCH_ESTIMATORS')))}

        random_search = RandomizedSearchCV(
            clf,
            param_distributions=param_distribution,
            n_iter=configparser.getint('RANDOM_SEARCH_COUNT'),
            n_jobs=configparser.getint('RANDOM_SEARCH_PROCESSES'),
            scoring=['roc_auc', 'f1', 'average_precision', 'accuracy'],
            cv=PredefinedSplit(test_fold),
            refit='average_precision')

        random_search.fit(all_x, all_y.ravel())

        return get_results_of_search(random_search.cv_results_)

    # read the previous prediction results
    pred_dic, train_x, train_y, predicates_train = features.get_x_y(
        configparser.getstr('TRAIN_DIR'),
        configparser.getstr('ER_MLP_MODEL_DIR'),
        configparser.getstr('PRA_MODEL_DIR'))

    pred_dic, test_x, test_y, predicates_test = features.get_x_y(
        configparser.getstr('DEV_DIR'),
        configparser.getstr('ER_MLP_MODEL_DIR'),
        configparser.getstr('PRA_MODEL_DIR'))

    # prediction results of adaboosst
    predictions_test = np.zeros_like(predicates_test, dtype=float)
    predictions_test = predictions_test.reshape((np.shape(predictions_test)[0], 1))

    model_dic = {}

    for key, idx in pred_dic.items():
        test_indices, = np.where(predicates_test == idx)

        # if we have matching indices
        if np.shape(test_indices)[0] != 0:
            test_x_predicate = test_x[test_indices]
            test_y_predicate = test_y[test_indices]

            train_indices, = np.where(predicates_train == idx)
            if np.shape(train_indices)[0] != 0:
                train_x_predicate = train_x[train_indices]
                train_y_predicate = train_y[train_indices]
            else:
                print('No training data for predicate: {}'.format(key))
                sys.exit()

            # use smote sampling to balance positives and negatives
            if configparser.getbool('USE_SMOTE_SAMPLING'):
                ros = SMOTE(sampling_strategy='minority')
                train_x_predicate, train_y_predicate = ros.fit_sample(train_x_predicate, train_y_predicate.ravel())
                train_y_predicate = train_y_predicate.reshape(-1, 1)

            if configparser.getbool('RUN_RANDOM_SEARCH'):
                ap_params = perform_randomized_search(train_x_predicate, train_y_predicate, test_x_predicate, test_y_predicate)
                configparser.append(
                    'RANDOM_SEARCH_BEST_PARAMS_{}'.format(key),
                    {'n_estimators': ap_params['n_estimators'],
                     'learning_rate': ap_params['learning_rate']})

            # build & fit model using the best parameters
            clf = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=1),
                n_estimators=configparser.getint('n_estimators', section='RANDOM_SEARCH_BEST_PARAMS_{}'.format(key)),
                learning_rate=configparser.getfloat('learning_rate', section='RANDOM_SEARCH_BEST_PARAMS_{}'.format(key)),
                random_state=0)

            if configparser.getbool('CLASSIFIER_CALIBRATION'):
                final_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
            else:
                final_clf = clf

            # fit the final chosen model (calibrated / uncalibrated)
            final_clf.fit(train_x_predicate, train_y_predicate.ravel())
            model_dic[idx] = final_clf

            # do prediction on the test set
            probs = final_clf.predict_proba(test_x_predicate)[:, 1]
            probs = np.reshape(probs, (-1, 1))
            predictions_test[test_indices] = probs[:]

    with open(os.path.join(model_save_dir, 'model.pkl'), 'wb') as output:
        pickle.dump(model_dic, output, pickle.HIGHEST_PROTOCOL)

    with open('./' + config_file, 'w') as configfile:
        configparser.write(configfile)

if __name__ == "__main__":
    main()
