"""
Filename: data_manager.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Python class to manage the dataset.
    Step 1. Perform name mapping.
    Step 2. Apply data rule to infer new data.

To-do:
    1. Maybe split integrate_data() into two functions name_map() and data_rule()
    2. Move _SPO_LIST to global file.
    3. Define remove temporal info predicates in the xml file.
"""
#!/usr/bin/python

import os
import sys
import logging as log
import xml.etree.ElementTree as ET
import pandas as pd

DIRECTORY = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(DIRECTORY, './'))

from utilities import get_pd_of_statement

class DataManager:
    """
    Class for managing the data.
    """

    #######################################
    # we probably need to get rid of this #
    #######################################
    SPO_LIST = ['Subject', 'Predicate', 'Object']
    CRA_STR = 'confers resistance to antibiotic'
    CNRA_STR = 'confers no resistance to antibiotic'
    UA_STR = 'upregulated by antibiotic'
    NUA_STR = 'not upregulated by antibiotic'

    def __init__(self, data_paths=None, map_file=None, data_rule_file=None):
        """
        Class constructor for DataManager. All inputs are optional
        since only name mapping may be used somewhere else.

        Inputs:
            data_paths: (optional) path & source for each dataset
            map_file: (optional) data name mapping file name
            data_rule_file: (optional) data rule file name
        """
        self.data_paths = data_paths
        self.map_file = map_file
        self.data_rule_file = data_rule_file

    def integrate_data(self):
        """
        Integrate data from multiple sources and
        generate name mapped, data rule applied data.

        Returns:
            pd_integrated_data: resulting integrated data

                       Subject     Predicate Object Source
                0          lrp  no represses   fadD  hiTRN
                1          fur  no represses   yfeD  hiTRN
                2          fnr  no represses   ybhN  hiTRN
                3          crp  no represses   uxuR  hiTRN
                                ...
        """
        list_integrated_data = []

        if isinstance(self.data_paths, str):
            pd_data_paths = pd.read_csv(self.data_paths, sep='\t', comment='#')
        else:
            pd_data_paths = self.data_paths

        # iterate over each dataset and perform name mappipng
        for _, row in pd_data_paths.iterrows():
            str_source = row['Source']
            str_path = row['Path']

            log.info('Processing source %s using %s', str_source, str_path)

            pd_data = pd.read_csv(str_path, '\t')

            # drop missing values if there's any
            before = pd_data.shape[0]
            pd_data = pd_data.dropna()
            after = pd_data.shape[0]

            if (before - after) > 0:
                log.warning('Dropping %d missing values.', (before - after))

            log.info('Applying name mapping to data from source %s', str_source)

            pd_data = self.name_map_data(pd_data)
            pd_data['Source'] = str_source
            list_integrated_data.append(pd_data)

            log.info('Added %d tuples from source %s', pd_data.shape[0], str_source)

        # apply data rule
        pd_integrated_data = pd.concat(list_integrated_data, sort=True)
        if self.data_rule_file:
            pd_integrated_data = self._apply_data_rule(pd_integrated_data)
        pd_integrated_data.index = range(pd_integrated_data.shape[0]) # update the index

        log.info('Total of %d tuples integrated.', pd_integrated_data.shape[0])

        return pd_integrated_data

    def drop_temporal_info(self, pd_data):
        """
        Remove temporal data in the predicate.

        Inputs:
            pd_data: integrated data using self.pd_data()

        Returns:
            pandas dataframe whose predicate does not contain temporal information
        """

        pd_data.loc[pd_data['Predicate'].str.startswith(self.CRA_STR), 'Predicate'] = self.CRA_STR
        pd_data.loc[pd_data['Predicate'].str.startswith(self.CNRA_STR), 'Predicate'] = self.CNRA_STR

        pd_data.loc[pd_data['Predicate'].str.startswith(self.UA_STR), 'Predicate'] = self.UA_STR
        pd_data.loc[pd_data['Predicate'].str.startswith(self.NUA_STR), 'Predicate'] = self.NUA_STR

        return pd_data.drop_duplicates()

    def name_map_data(self, pd_data):
        """
        (Private) Perform name mapping given data from single source.

        Inputs:
            pd_data: all the data from single source

        Returns:
            pd_converted_data: name mapped data
        """
        # open name mapping file
        with open(self.map_file) as _file:
            next(_file) # skip the header
            map_file_content = _file.readlines()

        # store dictionary of the name mapping information
        dict_map = {}
        for map_line in map_file_content:
            key, value = map_line.strip('\n').split('\t')
            dict_map[key] = value

        def has_mapping_name(row, dict_map):
            # return original if both subject and object are already using correct name
            if (row['Subject'] in dict_map.values()) and (row['Object'] in dict_map.values()):
                return row

            new_x = row.copy()

            if (row['Subject'] in dict_map) and (row['Subject'] not in dict_map.values()):
                new_x['Subject'] = dict_map[row['Subject']]

            if (row['Object'] in dict_map) and (row['Object'] not in dict_map.values()):
                new_x['Object'] = dict_map[row['Object']]

            return new_x

        pd_converted_data = pd_data.apply(has_mapping_name, axis=1, args=(dict_map, ))

        return pd_converted_data

    def _apply_data_rule(self, pd_data):
        """
        (Private) Apply data rule and infer new data.

        Inputs:
            pd_data: name mapped data integrated from all the sources

        Returns:
            pd_new_data: data with new inferred data added
        """
        data_rules = ET.parse(self.data_rule_file).getroot()
        pd_new_data = pd_data.copy()

        log.info('Applying data rule to infer new data using %s', self.data_rule_file)

        # iterate over each data rule
        for data_rule in data_rules:
            log.debug('Processing data rule %s', data_rule.get('name'))

            # find all the triples that meets the data rule's if statement
            if_statement = data_rule.find('if')
            pd_if_statement = get_pd_of_statement(if_statement)
            indices_meeting_if = (pd_data[pd_if_statement.index] == pd_if_statement).all(1).values
            pd_rule_specific_new_data = pd_data[indices_meeting_if].copy()

            if pd_rule_specific_new_data.shape[0] == 0:
                log.debug('\tNo new data inferred using this data rule')
                continue

            # get the then statement that we are going to apply
            # and change it with the original (predicate, object, or both)
            then_statement = data_rule.find('then')
            pd_then_statement = get_pd_of_statement(then_statement)
            pd_rule_specific_new_data[pd_then_statement.index] = pd_then_statement.tolist()

            # append the new data to the original
            pd_new_data = pd_new_data.append(pd_rule_specific_new_data)

            log.debug('\t%d new tuples found using this data rule',
                      pd_rule_specific_new_data.shape[0])

        # drop the duplicates
        pd_new_data = pd_new_data.drop_duplicates()

        log.info('Total of %d new tuples added based on the data rule after dropping the duplicates',
                 (pd_new_data.shape[0] - pd_data.shape[0]))

        return pd_new_data
