"""
Filename: create_kg.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu
    Jason Youn -jyoun@ucdavis.edu

Description:
    Create the knowledge graph.

To-do:
"""

# standard imports
import argparse
import logging as log

# third party imports
import pandas as pd

# local imports
from integrate_modules.data_manager import DataManager
from integrate_modules.inconsistency_manager import InconsistencyManager
from integrate_modules.report_manager import plot_trustworthiness

# default arguments
DEFAULT_DATA_PATH_STR = './data/data_path_file.txt'
DEFAULT_MAP_STR = './data/data_map.txt'
DEFAULT_DATA_RULE_STR = './data/data_rules.xml'
DEFAULT_INCONSISTENCY_RULES_STR = './data/inconsistency_rules.xml'
DEFAULT_WITHOUT_INCONSISTSENCIES_STR = './output/kg_without_inconsistencies.txt'
DEFAULT_RESOLVED_INCONSISTENCIES_STR = './output/resolved_inconsistencies.txt'
DEFAULT_VALIDATION_FILE_TXT = './data/inconsistency_validation/validation_results.txt'
DEFAULT_VALIDATED_INCONSISTENCIES_STR = './output/validated_inconsistencies.txt'
DEFAULT_FINAL_KG_STR = './output/kg_final.txt'
DEFAULT_PHASE_STR = 'all'

def set_logging():
    """
    Configure logging.
    """
    log.basicConfig(format='(%(levelname)s) %(filename)s: %(message)s', level=log.DEBUG)

    # set logging level to WARNING for matplotlib
    logger = log.getLogger('matplotlib')
    logger.setLevel(log.WARNING)

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Integrate knowledgebase from multiple sources.')

    parser.add_argument(
        '--data_path',
        default=DEFAULT_DATA_PATH_STR,
        help='Path to the file data_path_file.txt')

    parser.add_argument(
        '--map',
        default=DEFAULT_MAP_STR,
        help='Path to the file data_map.txt')

    parser.add_argument(
        '--data_rule',
        default=DEFAULT_DATA_RULE_STR,
        help='Path to the file data_rules.xml')

    parser.add_argument(
        '--inconsistency_rule',
        default=DEFAULT_INCONSISTENCY_RULES_STR,
        help='Path to the file inconsistency_rules.xml')

    parser.add_argument(
        '--without_inconsistsencies',
        default=DEFAULT_WITHOUT_INCONSISTSENCIES_STR,
        help='Path to save the knowledge graph without inconsistencies')

    parser.add_argument(
        '--resolved_inconsistencies',
        default=DEFAULT_RESOLVED_INCONSISTENCIES_STR,
        help='Path to save the inconsistencies file')

    parser.add_argument(
        '--validation_file',
        default=DEFAULT_VALIDATION_FILE_TXT,
        help='Filepath containing validation info')

    parser.add_argument(
        '--validated_inconsistencies',
        default=DEFAULT_VALIDATED_INCONSISTENCIES_STR,
        help='Path for validationed inconsistencies file')

    parser.add_argument(
        '--data_out',
        default=DEFAULT_FINAL_KG_STR,
        help='Path to save the final knowledge graph')

    parser.add_argument(
        '--use_temporal',
        default=False,
        action='store_true',
        help='Remove temporal data unless this option is set')

    parser.add_argument(
        '--phase',
        default=DEFAULT_PHASE_STR,
        help='Select one of three phase strings (phase1 | phase2 | phase3 | all)')

    # check for correct phase argument
    phase_arg = parser.parse_args().phase
    if phase_arg not in ['phase1', 'phase2', 'phase3', 'all']:
        raise ValueError('Invalid phase argumnet \'{}\'!'.format(phase_arg))

    return parser.parse_args()

def main():
    """
    Main function.
    """
    # set log and parse args
    set_logging()
    args = parse_argument()

    # construct DataManager class
    data_manager = DataManager(args.data_path, args.map, args.data_rule)

    # construct InconsistencyManager class
    inconsistency_manager = InconsistencyManager(
        args.inconsistency_rule, resolver_mode='AverageLog')

    if args.phase == 'phase1':
        # perform 1) knowledge integration and 2) knowledge rule application
        pd_data = data_manager.integrate_data()

        # remove temporal data in predicate
        if not args.use_temporal:
            pd_data = data_manager.drop_temporal_info(pd_data)

        # perform inconsistency detection
        inconsistencies = inconsistency_manager.detect_inconsistencies(pd_data)

        # perform inconsistency resolution and parse the results
        resolution_result = inconsistency_manager.resolve_inconsistencies(pd_data, inconsistencies)
        pd_resolved_inconsistencies = resolution_result[0]
        pd_without_inconsistencies = resolution_result[1]
        np_trustworthiness_vector = resolution_result[2]

        # report data integration results
        plot_trustworthiness(pd_data, np_trustworthiness_vector, inconsistencies)

        # save results for wet lab validation
        log.info('Saving knowledge graph without inconsistencies to \'%s\'',
                 args.without_inconsistsencies)
        pd_without_inconsistencies.to_csv(args.without_inconsistsencies, index=False, sep='\t')

        log.info('Saving resolved inconsistencies to \'%s\'', args.resolved_inconsistencies)
        pd_resolved_inconsistencies.to_csv(args.resolved_inconsistencies, index=False, sep='\t')

    if args.phase == 'phase2':
        pd_resolved = pd.read_csv(args.resolved_inconsistencies, sep='\t')
        pd_validated = pd.read_csv(args.validation_file, sep='\t')
        pd_resolved = data_manager.name_map_data(pd_resolved)

        inconsistency_manager.validate_resolved_inconsistencies(
            pd_resolved,
            pd_validated,
            args.validated_inconsistencies)

    if args.phase == 'phase3':
        # load previously saved results and validation results
        pd_without_inconsistencies = pd.read_csv(args.without_inconsistsencies, sep='\t')
        pd_validated_inconsistencies = pd.read_csv(
            args.validated_inconsistencies,
            sep='\t',
            na_values=[],
            keep_default_na=False)

        # insert resolved inconsistsencies back into the KG
        pd_data_final = inconsistency_manager.reinstate_resolved_inconsistencies(
            pd_without_inconsistencies,
            pd_validated_inconsistencies)

        # save integrated data
        pd_data_final.to_csv(args.data_out, index=False, sep='\t')

if __name__ == '__main__':
    main()
