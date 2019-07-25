"""
Filename: figures.py

Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Draw figures.

To-do:
"""
# standard imports
import argparse
import os
import sys

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# global variables
DEFAULT_OUTDIR_STR = '../output'
DEFAULT_CONFIDENCE_FILE_STR = 'hypotheses_confidence.txt'

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Analyze hypotheses.')

    parser.add_argument(
        '--outdir',
        default=DEFAULT_OUTDIR_STR,
        help='Output directory.')

    return parser.parse_args()

def bin_hypotheses(pd_data):
    intervals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    intervals.reverse()

    dict_bin_count = {}

    for i, interval in enumerate(intervals):
        if interval == 0.9:
            bin_name = '[{}, {}]'.format(interval, 1)
        else:
            bin_name = '[{}, {})'.format(interval, intervals[i-1])

        pd_bin = pd_data[pd_data['Confidence'] >= interval]
        dict_bin_count[bin_name] = pd_bin.shape[0]
        pd_data = pd_data.drop(pd_bin.index)

    print(dict_bin_count)

    list_values = list(dict_bin_count.values())
    list_values.reverse()

    list_keys = list(dict_bin_count.keys())
    list_keys.reverse()

    fig, ax = plt.subplots()
    bp = ax.bar(np.arange(len(dict_bin_count)), list_values, log=True)

    ax.set_xlabel('Confidence interval')
    ax.set_ylabel('Number of hypotheses (log)')
    ax.set_xticks(np.arange(len(dict_bin_count)))
    ax.set_xticklabels(list_keys, rotation=45)

    plt.tight_layout()

def which_to_test(pd_data):
    confidence = 0.7

    pd_filtered = pd_data[pd_data['Confidence'] >= confidence]
    print(pd_filtered.shape[0])

    pd_group = pd_filtered.groupby('Object').size()
    pd_group = pd_group.sort_values(ascending=False).to_frame(name='count')
    pd_group['cum_sum'] = pd_group['count'].cumsum()
    pd_group['cum_percentage'] = (pd_group['cum_sum'] / pd_group['count'].sum()) * 100

    print(pd_group)

    # pd_group.reset_index().to_csv('~/Jason/Shared/to_test.txt', sep='\t')

def main():
    """
    Main function.
    """
    args = parse_argument()

    filepath = os.path.join(args.outdir, DEFAULT_CONFIDENCE_FILE_STR)
    col_names = ['Subject', 'Predicate', 'Object', 'Label', 'Confidence']
    pd_data = pd.read_csv(filepath, sep='\t', names=col_names)

    # bin_hypotheses(pd_data.copy())
    which_to_test(pd_data.copy())

    # plt.show()

if __name__ == '__main__':
    main()
