"""
Filename: analyze_inconsistency_validation.py

Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:

To-do:
"""

import os
import sys
import argparse
import logging as log
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix

ABS_PATH_METRICS = os.path.join(os.path.dirname(__file__), '../integrate_modules')
sys.path.insert(0, ABS_PATH_METRICS)

from data_manager import DataManager

pd.options.mode.chained_assignment = None

# default paths
DEFAULT_OUTPUT_DIR_STR = '../output'

# default file names
DEFAULT_INCONSISTENCY_FILE_TXT = '../output/resolved_inconsistencies.txt'
DEFAULT_VALIDATION_FILE_TXT = '../data/inconsistency_validation/validation_results.txt'
DEFAULT_ANALYSIS_RESULT_TXT = 'inconsistency_analysis.txt'
DEFAULT_COMBINED_EXCEL = 'resolution_validation_combined.xlsx'

DEFAULT_MAP_FILE = '../data/data_map.txt'

CRA_STR = 'confers resistance to antibiotic'
CNRA_STR = 'confers no resistance to antibiotic'

def set_logging():
    """
    Configure logging.
    """
    log.basicConfig(format='(%(levelname)s) %(filename)s: %(message)s', level=log.DEBUG)

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Analyze inconsistency validation experimental results.')

    parser.add_argument(
        '--output_path',
        default=DEFAULT_OUTPUT_DIR_STR,
        help='Directory to save the analysis results')

    parser.add_argument(
        '--inconsistency_file',
        default=DEFAULT_INCONSISTENCY_FILE_TXT,
        help='Filepath containing resolved inconsistencies info')

    parser.add_argument(
        '--validation_file',
        default=DEFAULT_VALIDATION_FILE_TXT,
        help='Filepath containing validation info')

    parser.add_argument(
        '--threshold_key',
        default=None,
        help='String of column name to use for thresholding')

    parser.add_argument(
        '--save_only_validated',
        default=False,
        action='store_true',
        help='Remove temporal data unless this option is set')

    return parser.parse_args()

def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0

    return tp / (tp + fn)

def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0

    return tp / (tp + fp)

def calculate_f1(recall, precision):
    if precision + recall == 0:
        return 0

    return (2 * precision * recall) / (precision + recall)

def combine_resolution_with_validation(resolved, validated, save_only_validated, save_path=None):
    if Path(resolved).suffix == '.txt':
        pd_combined = pd.read_csv(resolved, sep='\t')
    elif Path(resolved).suffix == '.xlsx':
        pd_combined = pd.read_excel(resolved)
    else:
        sys.exit('Invalid file extension for given file: {}'.format(resolved))

    if Path(validated).suffix == '.txt':
        pd_validate = pd.read_csv(validated, sep='\t')
    elif Path(validated).suffix == '.xlsx':
        pd_validate = pd.read_excel(validated)
    else:
        sys.exit('Invalid file extension for given file: {}'.format(validated))

    # need to perform name mapping because we may compare
    # validation with only single source that has no name mapping yet
    pd_combined = DataManager(map_file=DEFAULT_MAP_FILE).name_map_data(pd_combined, resolved)

    # add new columns that will be used for validation
    pd_combined['Validation'] = ''
    pd_combined['Match'] = ''

    for _, row in pd_validate.iterrows():
        gene = row.Subject
        predicate = row.Predicate
        antibiotic = row.Object

        match = pd_combined[pd_combined.Subject == gene]
        match = match[match.Object == antibiotic]

        # if there are more than one matches, there is something wrong
        if match.shape[0] > 1:
            log.error('Found {} matching resolved inconsistencies for ({}, {}).'.format(match.shape[0], gene, antibiotic))
            sys.exit()

        if match.shape[0] == 0:
            continue

        pd_combined.loc[match.index, 'Validation'] = predicate

        if pd_combined.loc[match.index, 'Predicate'].str.contains(predicate).values[0]:
            pd_combined.loc[match.index, 'Match'] = 'True'
        else:
            pd_combined.loc[match.index, 'Match'] = 'False'

    print(pd_combined)

    if save_path:
        if save_only_validated:
            pd_combined = pd_combined[pd_combined.Match != '']

        pd_combined.to_excel(save_path, index=False)

    return pd_combined

def calculate_statistics(pd_data, threshold_key=None):
    # only look at the data that we validated
    pd_data = pd_data[pd_data.Match != '']

    pd_data.loc[:, 'Resolution label'] = 0
    idx = pd_data['Predicate'].str.match(CRA_STR)
    pd_data.loc[idx, 'Resolution label'] = 1

    pd_data.loc[:, 'Validation label'] = 0
    idx = pd_data['Validation'].str.match(CRA_STR)
    pd_data.loc[idx, 'Validation label'] = 1

    cm_result = confusion_matrix(pd_data['Validation label'], pd_data['Resolution label'])

    print('-----')
    print(cm_result[1, 1], cm_result[0, 1])
    print(cm_result[1, 0], cm_result[0, 0])
    print('-----')

    if threshold_key:
        threshold_param = pd_data[threshold_key]
        param_unique = np.sort(threshold_param.unique())

        param_list = []
        f1_list = []
        for param in param_unique:
            pd_pass = pd_data[pd_data[threshold_key] >= param]
            cm_result = confusion_matrix(pd_pass['Validation label'], pd_pass['Resolution label'])

            if cm_result.shape == (1, 1):
                log.warning('Confusion matrix output has size {} for {} {}.'.format(cm_result.shape, threshold_key, param))
                continue

            tp = cm_result[1, 1]
            fp = cm_result[0, 1]
            fn = cm_result[1, 0]
            tn = cm_result[0, 0]

            print('-----')
            print(tp, fp)
            print(fn, tn)
            print('-----')

            recall = calculate_recall(tp, fn)
            precision = calculate_precision(tp, fp)
            f1 = calculate_f1(recall, precision)

            param_list.append(param)
            f1_list.append(f1)

            print(param, f1, precision, recall)

        print(param_list)
        print(f1_list)

        plt.figure()
        plt.plot(param_list, f1_list)

def main():
    """
    Main function.
    """
    # set log and parse args
    set_logging()
    args = parse_argument()

    pd_combined = combine_resolution_with_validation(
        args.inconsistency_file,
        args.validation_file,
        args.save_only_validated,
        save_path=os.path.join(args.output_path, DEFAULT_COMBINED_EXCEL))

    calculate_statistics(pd_combined.copy(), threshold_key=args.threshold_key)

    plt.show()


if __name__ == '__main__':
    main()
