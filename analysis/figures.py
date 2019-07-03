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
import glob
import os
import sys

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp

# global variables
DEFAULT_OUTDIR_STR = '../output'
DEFAULT_CONFIDENCE_FILE_STR = 'hypotheses_confidence.txt'

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

def analyze_hypotheses(filepath):
    col_names = ['Subject', 'Predicate', 'Object', 'Label', 'Confidence']
    pd_data =pd.read_csv(filepath, sep='\t', names=col_names)

    # classify confidence into bins
    bins = np.linspace(0, 1, 11)
    pd_data['bin'] = pd.cut(pd_data['Confidence'], bins, include_lowest=True)

    pd_data['bin'] = pd_data['bin'].astype(str)

    bin_size = pd_data.groupby(pd_data['bin']).size()
    bin_size = bin_size.rename(index={'(-0.001, 0.1]': '[0.0, 0.1]'})

    print(bin_size)

    ax = bin_size.plot.bar(logy=True, rot=45)

    plt.title('Number of hypotheses belonging to each confidence interval.')
    plt.xlabel('Confidence interval')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function.
    """
    args = parse_argument()

    analyze_hypotheses(os.path.join(args.outdir, DEFAULT_CONFIDENCE_FILE_STR))

if __name__ == '__main__':
    main()
