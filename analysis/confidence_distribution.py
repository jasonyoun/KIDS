import argparse
from ast import literal_eval
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_BASE_DIR = '../stacked/model_instance'

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Analyze confidence distribution.')

    parser.add_argument(
        '--dir',
        default=DEFAULT_BASE_DIR,
        help='Stacked results base directory.')

    return parser.parse_args()

def main():

    args = parse_argument()

    interval = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    dict_interval = dict([(key, []) for key in interval])

    # for each interval, find the percentage of triplets belonging
    # to that interval that actually has label 1
    for fold in range(5):
        fold_dir = os.path.join(args.dir, 'fold_{}/test'.format(fold))

        pd_classification = pd.read_csv(
            os.path.join(fold_dir, 'classifications_stacked.txt'),
            names=['classification'],
            header=None)

        pd_confidence = pd.read_csv(
            os.path.join(fold_dir, 'confidence_stacked.txt'),
            names=['confidence'],
            header=None)

        pd_data = pd.concat([pd_classification, pd_confidence], axis=1)

        pd_data['confidence'] = pd_data['confidence'].apply(literal_eval)
        pd_data['confidence'] = pd_data['confidence'].apply(lambda x: x[0])

        pd_data['label'] = pd_data.index
        pd_data['label'] = pd_data['label'] % 50
        pd_data['label'] = (pd_data['label'] == 0)

        pd_copy = pd_data.copy()
        for i in interval:
            pd_filtered = pd_copy[pd_copy['confidence'] >= i]

            pd_one = pd_filtered[pd_filtered['label']]
            pd_zero = pd_filtered[~pd_filtered['label']]

            dict_interval[i].append(pd_one.shape[0] / pd_filtered.shape[0])

            pd_copy = pd_copy.drop(pd_filtered.index)

    print(dict_interval)

    #
    interval.reverse()
    list_mean = []
    list_std = []

    for i in interval:
        list_mean.append(np.mean(dict_interval[i]))
        list_std.append(np.std(dict_interval[i]))


    fig, ax = plt.subplots()

    bp = ax.bar(np.arange(len(interval)), list_mean, yerr=list_std)

    # ax.set_title('Number of edges vs. F1')
    # ax.set_xlabel('Number of edges bin\n(Number of samples in each bin)')
    # ax.set_ylabel('F1 score')
    # ax.set_xticks(ind)
    # ax.set_xticklabels(xticklabels_list)

    plt.show()

if __name__ == '__main__':
    main()
