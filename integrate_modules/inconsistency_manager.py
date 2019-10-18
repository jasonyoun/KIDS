"""
Filename: inconsistency_manager.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu

Description:
    Detect the inconsistencies.

To-do:
    1. Move _SPO_LIST to global file.
    2. Move inconsistency correctors as sub class.
"""

# standard imports
import logging as log

# third party imports
import pandas as pd
import xml.etree.ElementTree as ET

# local imports
from .utilities import get_pd_of_statement
from .inconsistency_correctors.inconsistency_corrector import InconsistencyCorrector

class InconsistencyManager(InconsistencyCorrector):
    """
    Class for detecting the inconsistencies.
    """
    _SPO_LIST = ['Subject', 'Predicate', 'Object']

    def __init__(self, inconsistency_rule_file, resolver_mode='AverageLog'):
        """
        Class constructor foro InconsistencyManager.

        Inputs:
            inconsistency_rule_file: XML file name containing the inconsistency rules
            resolver_mode: string denoting which inconsistency resolution to use
                (AverageLog | Investment | PooledInvestment | Sums | TruthFinder | Voting)
        """
        self.inconsistency_rule_file = inconsistency_rule_file

        # inherit InconsistencyCorrector
        self.inconsistency_corrector = InconsistencyCorrector(resolver_mode)

    def detect_inconsistencies(self, pd_data):
        """
        Detect the inconsistencies among the data using the provided rule file.

        Inputs:
            pd_data: data to detect inconsistency from

                           Subject     Predicate Object Source
                    0          lrp  no represses   fadD  hiTRN
                    1          fur  no represses   yfeD  hiTRN
                    2          fnr  no represses   ybhN  hiTRN
                    3          crp  no represses   uxuR  hiTRN

        Returns:
            inconsistencies: Dictionary containing inconsistency_id as key
                and list of inconsistent triples + source as value

                {
                0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
                    (('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
                1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
                    ('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
                                        ...
                }
        """
        inconsistency_rules = ET.parse(self.inconsistency_rule_file).getroot()
        inconsistencies = {}
        inconsistency_id = 0

        log.info('Detecting inconsistencies using inconsisty rule %s', self.inconsistency_rule_file)

        # iterate over each inconsistency rule
        for inconsistency_rule in inconsistency_rules:
            inconsistency_rule_name = inconsistency_rule.get('name')
            condition_statement = inconsistency_rule.find('condition')

            log.debug('Processing inconsisency rule \'%s\'', inconsistency_rule_name)

            # check if condition is met
            pd_condition_specific_data = pd_data.copy()

            if condition_statement is not None:
                pd_condition_statement = get_pd_of_statement(condition_statement)
                indices_meeting_condition = (pd_data[pd_condition_statement.index] ==
                                             pd_condition_statement).all(1).values
                pd_condition_specific_data = pd_data[indices_meeting_condition].copy()

                # skip to the next rule if condition statement
                # exists and there are no data meeting the condition
                if pd_condition_specific_data.shape[0] == 0:
                    log.debug('Skipping \'%s\' because there are no data meeting condition',
                              inconsistency_rule_name)
                    continue

            # check if data has conflicts
            inconsistency_statement = inconsistency_rule.find('inconsistency')
            conflict_feature_name = inconsistency_statement.get('name')
            conflict_feature_values = [inconsistency_feature.get('value')
                                       for inconsistency_feature in inconsistency_statement]

            # skip to the next rule if there are no conflicts
            if not self._data_has_conflict_values(
                    pd_data[conflict_feature_name], conflict_feature_values):
                log.debug('Skipping \'%s\' because there are no conflicts', inconsistency_rule_name)
                continue

            rest_feature_names = [feature_name
                                  for feature_name in self._SPO_LIST
                                  if feature_name != conflict_feature_name]
            pd_grouped_data = pd_data.groupby(rest_feature_names)[conflict_feature_name].apply(set)

            def has_conflict_values(data, conflict_feature_values):
                return data.intersection(conflict_feature_values)

            pd_nconflict_data = pd_grouped_data.apply(
                has_conflict_values, args=(set(conflict_feature_values), ))
            pd_filtered_data = pd_nconflict_data[pd_nconflict_data.apply(len) > 1]

            # create inconsistency triple list
            for row_idx in range(pd_filtered_data.shape[0]):
                if row_idx % 100 == 0:
                    log.debug('Creating list of inconsistencies: %d/%d',
                              row_idx, pd_filtered_data.shape[0])

                pd_conflict_data = pd.Series(pd_filtered_data.index[row_idx],
                                             index=rest_feature_names)

                conflict_tuples = []
                for conflict_value in pd_filtered_data[row_idx]:
                    pd_conflict_data[conflict_feature_name] = conflict_value
                    sources = pd.unique(
                        pd_condition_specific_data[(pd_condition_specific_data[self._SPO_LIST]
                                                    == pd_conflict_data).all(1)]['Source'])
                    conflict_tuples.append((tuple(pd_conflict_data[self._SPO_LIST]),
                                            sources.tolist()))

                inconsistencies[inconsistency_id] = conflict_tuples
                inconsistency_id = inconsistency_id + 1

        log.info('Found %d inconsistencies', len(inconsistencies))

        return inconsistencies

    def resolve_inconsistencies(self, pd_data, inconsistencies, **kwargs):
        """
        Wrapper function for inconsistency resolver.

        Inputs:
            pd_data: dataframe of integrated data that has inconsistencies to resolve
            inconsistencies: detected inconsistencies
            kwargs: arguments specific to algorithm to use
                    (refer to each resolution algorithm)

        Returns:
            (pd_resolved_inconsistencies, pd_without_inconsistencies, np_trustworthiness_vector)
        """
        return self.inconsistency_corrector.resolve_inconsistencies(
            pd_data, inconsistencies, **kwargs)

    @staticmethod
    def validate_resolved_inconsistencies(
            pd_resolved,
            pd_validated,
            save_path,
            save_only_validated=False):

        # add new columns that will be used for validation
        pd_combined = pd_resolved.copy()
        pd_combined['Validation'] = ''
        pd_combined['Match'] = ''

        for _, row in pd_validated.iterrows():
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

        if save_only_validated:
            pd_combined = pd_combined[pd_combined.Match != '']

        pd_combined.to_csv(save_path, sep='\t', index=False)

        return pd_combined

    @staticmethod
    def reinstate_resolved_inconsistencies(
            pd_data_without_inconsistencies,
            pd_validated,
            mode='only_validated'):
        """
        Insert resolved inconsistencies back into the
        original data to create final knowledge graph.

        Inputs:
            pd_data_without_inconsistencies: data without inconsistent triplets
            pd_validated: inconsistencies where some / all of them are validated
            mode: which resolved inconsistencies to reinstate ('all' | 'only_validated')

        Returns:
            pd_data_final: dataframe containing resolved inconsistencies
        """
        # column names to use
        final_column_names = ['Subject', 'Predicate', 'Object', 'Belief', 'Source size', 'Sources']
        columns_to_drop = ['Total source size', 'Mean belief of conflicting tuples',
                           'Conflicting tuple info']

        # drop unnecessary columns
        pd_filtered = pd_validated.drop(columns=columns_to_drop)

        log.info('Number of data without inconsistencies: %d',
                 pd_data_without_inconsistencies.shape[0])
        log.info('Number of resolved inconsistencies: %d', pd_filtered.shape[0])

        # select triplets to append depending on the selected mode
        if mode == 'all':
            pd_to_append = pd_filtered
        elif mode == 'only_validated':
            pd_to_append = pd_filtered[pd_filtered['Validation'] != '']
            pd_to_append = pd_to_append[pd_to_append['Match'] == 'True']
        else:
            raise ValueError('Invalid mode \'{}\' passed'.format(mode))

        log.info('Number of resolved inconsistencies to append: %d', pd_to_append.shape[0])

        # append the selected triplets
        pd_to_append = pd_to_append.loc[:, final_column_names]
        pd_data_final = pd.concat([pd_data_without_inconsistencies, pd_to_append],
                                  ignore_index=True, sort=False)

        log.info('Number of triplets in the final knowledge graph: %d', pd_data_final.shape[0])

        return pd_data_final

    @staticmethod
    def _data_has_conflict_values(all_feature_values, conflict_feature_values):
        """
        (Private) Check is data has conflicting values.

        Inputs:
            all_feature_values: column from one of Subject/Predicate/Object
            conflict_feature_values: list containing the conflicting feature values
                ['response to antibiotic', 'no response to antibiotic']

        Returns:
            True if data has conflicting values, False otherwise.
        """
        unique_feature_values = pd.unique(all_feature_values)
        num_of_conflict_values = 0

        for conflict_feature_value in conflict_feature_values:
            if conflict_feature_value in unique_feature_values:
                num_of_conflict_values = num_of_conflict_values + 1

        return num_of_conflict_values > 1
