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
DEFAULT_TEST_STATS_FILE_STR = 'test_stats.txt'

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Draw figures.')

    parser.add_argument(
        '--outdir',
        default=DEFAULT_OUTDIR_STR,
        help='Output directory.')

    return parser.parse_args()

def edges_statistics(filepath):
    pd_data = pd.read_csv(filepath, sep='\t')
    pd_data['bin'] = pd.qcut(pd_data['edges_in_train'], q=5)
    pd_data['bin_right_end'] = pd_data['bin'].apply(lambda x: x.right)
    pd_data['bin'] = pd_data['bin'].astype(str)
    pd_data['bin'] = pd_data['bin'].replace('\(-0\.001', '[0.0', regex=True)
    pd_data['bin'] = pd_data['bin'].replace('\.0', '', regex=True)

    # variables for boxplot
    pra_data_list = []
    mlp_data_list = []
    stacked_data_list = []
    xticklabels_list = []

    for _, val in enumerate(np.unique(pd_data['bin_right_end'].tolist())):
        pd_same_bin = pd_data[pd_data['bin_right_end'] == val].reset_index(drop=True)

        pra_data_list.append(pd_same_bin['pra_f1'])
        mlp_data_list.append(pd_same_bin['er_f1'])
        stacked_data_list.append(pd_same_bin['stacked_f1'])
        xticklabels_list.append('{}\n({})'.format(pd_same_bin['bin'][0], pd_same_bin.shape[0]))

    pd_bin_values = pd_data['bin'].value_counts()

    # draw boxplot
    ind = np.arange(pd_bin_values.shape[0])
    width = 0.2

    fig, ax = plt.subplots()

    bp1 = ax.boxplot(pra_data_list, positions=ind - width, widths=width, patch_artist=True)
    bp2 = ax.boxplot(mlp_data_list, positions=ind, widths=width, patch_artist=True)
    bp3 = ax.boxplot(stacked_data_list, positions=ind + width, widths=width, patch_artist=True)

    ax.set_title('Number of edges vs. F1')
    ax.set_xlabel('Number of edges bin\n(Number of samples in each bin)')
    ax.set_ylabel('F1 score')
    ax.set_xticks(ind)
    ax.set_xticklabels(xticklabels_list)

    # box plot 1 patch_artist
    for box in bp1['boxes']:
        box.set(facecolor='#e53935')
    for median in bp1['medians']:
        median.set(color='#000000')

    # box plot 2 patch_artist
    for box in bp2['boxes']:
        box.set(facecolor='#00897b')
    for median in bp2['medians']:
        median.set(color='#000000')

    # box plot 3 patch_artist
    for box in bp3['boxes']:
        box.set(facecolor='#8e24aa')
    for median in bp3['medians']:
        median.set(color='#000000')

    ax.legend((bp1['boxes'][0], bp2['boxes'][0], bp3['boxes'][0]), ('PRA', 'MLP', 'Stacked'), loc='upper left')

    plt.tight_layout()

def main():
    """
    Main function.
    """
    args = parse_argument()

    edges_statistics(os.path.join(args.outdir, DEFAULT_TEST_STATS_FILE_STR))

    plt.show()

if __name__ == '__main__':
    main()
