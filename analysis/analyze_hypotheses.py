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

def bin_hypotheses(filepath):
    col_names = ['Subject', 'Predicate', 'Object', 'Label', 'Confidence']
    pd_data = pd.read_csv(filepath, sep='\t', names=col_names)

    intervals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    intervals.reverse()

    dict_bin_count = {}
    pd_copy = pd_data.copy()

    for i, interval in enumerate(intervals):
        if interval == 0.9:
            bin_name = '[{}, {}]'.format(interval, 1)
        else:
            bin_name = '[{}, {})'.format(interval, intervals[i-1])

        pd_bin = pd_copy[pd_copy['Confidence'] >= interval]
        dict_bin_count[bin_name] = pd_bin.shape[0]
        pd_copy = pd_copy.drop(pd_bin.index)

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

def main():
    """
    Main function.
    """
    args = parse_argument()

    bin_hypotheses(os.path.join(args.outdir, DEFAULT_CONFIDENCE_FILE_STR))

    plt.show()

if __name__ == '__main__':
    main()
