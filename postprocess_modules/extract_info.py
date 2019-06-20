"""
Filename: extract_info.py

Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Given a integrated knowledge base, extract information
    like entities / relations and create set of functions
    to access these information.

To-do:
"""
import logging as log
import numpy as np
import pandas as pd

class ExtractInfo():
    """
    Class for extracting information from the integrated dataset.
    """

    # class variables (need to be removed later)
    SUBJECT_STR = 'Subject'
    PRED_STR = 'Predicate'
    OBJECT_STR = 'Object'

    DOMAIN_STR = 'Domain'
    RELATION_STR = 'Relation'
    RANGE_STR = 'Range'

    def __init__(self, pd_data, dr_path):
        """
        Class constructor for ExtractInfo.

        Inputs:
            pd_data: integrated data where
                pd_data.columns.values = ['Subject' 'Predicate' 'Object' 'Label']
            dr_path: path to the text file which contains the
                relation / domain / range relationship
        """
        self.pd_data = pd_data
        self.pd_dr, self.relations, self.entity_types = self._read_dr(dr_path)
        self.entity_dic = self._fill_entity_dic()

    def get_entity_by_type(self, entity_type):
        """
        Get set of entities by specifying the type of entity.

        Inputs:
            entity_type: type of entity (e.g. 'gene', 'antibiotic')

        Returns:
            numpy array containing the entities matching the type
        """
        assert entity_type in list(self.entity_dic.keys())

        return self.entity_dic[entity_type]

    def save_all_entities(self, file_path):
        """
        Save all the entities to the specified file location.

        Inputs:
            file_path: file path to save all the entities
        """
        log.info('Saving entities to \'%s\'...', file_path)

        all_entities = np.array([])

        for entity_type in list(self.entity_dic.keys()):
            all_entities = np.append(all_entities, self.entity_dic[entity_type])

        all_entities = np.unique(all_entities)

        np.savetxt(file_path, all_entities, fmt='%s')

    def save_entity_full_names(self, file_path):
        """
        Save entity full names to the specified file location.
        ex) concept:gene:gene_type

        Inputs:
            file_path: file path to save the entity full names to
        """
        log.info('Saving entity full names to \'%s\'...', file_path)

        entity_full_names = np.array([])

        for entity_type in list(self.entity_dic.keys()):
            pd_entities_for_type = pd.DataFrame(np.copy(self.entity_dic[entity_type]))
            pd_entities_for_type.iloc[:, 0] = 'concept:{}:'.format(entity_type) + pd_entities_for_type.iloc[:, 0].astype(str)
            entity_full_names = np.append(entity_full_names, pd_entities_for_type.values.ravel())

        np.savetxt(file_path, entity_full_names, fmt='%s')

    def save_hypotheses(self, file_path, relations):
        """
        Save hypotheses to generate hypothesis on.

        Inputs:
            file_path: path to save the hypotheses
            relations: list of strings of relations
                       e.g. ['represses', 'confers resistance to antibiotic']
        """
        pd_hypotheses = pd.DataFrame(columns=['Subject', 'Predicate', 'Object'])

        for relation in relations:
            log.info('Processing hypotheses for relation \'%s\'...', relation)

            # extract domain and range type for chosen relation
            row = self.pd_dr[self.pd_dr['Relation'] == relation]
            np_domain_entities = self.get_entity_by_type(row['Domain'].iloc[0])
            np_range_entities = self.get_entity_by_type(row['Range'].iloc[0])

            # generate list of tuples (sub, obj) from known triplets
            pd_known = self.pd_data[self.pd_data['Predicate'] == relation]
            known_sub_obj_tuple_list = [tuple(x) for x in pd_known[['Subject', 'Object']].values]

            # generate list of tuples (sub, obj) of all possible combinations
            all_sub_obj_tuple_list = np.array(np.meshgrid(np_domain_entities, np_range_entities)).T.reshape(-1, 2)
            all_sub_obj_tuple_list = list(map(tuple, all_sub_obj_tuple_list))

            log.debug('Size of all possible combinations of subject and object: %d', len(all_sub_obj_tuple_list))
            log.debug('Size of known combinations of subject and object: %d', len(known_sub_obj_tuple_list))

            # generate list of tuples (sub, obj) of unknown combinations
            unknown_combinations = list(set(all_sub_obj_tuple_list) - set(known_sub_obj_tuple_list))

            log.debug('Size of unknown combinations of subject and object: %d', len(unknown_combinations))

            # append the unknown triplets to generate hypothesis on
            pd_hypotheses_to_append = pd.DataFrame(unknown_combinations, columns=['Subject', 'Object'])
            pd_hypotheses_to_append.insert(1, column='Predicate', value=relation)
            pd_hypotheses = pd_hypotheses.append(pd_hypotheses_to_append, sort=False)

        pd_hypotheses.to_csv(file_path, sep='\t', index=False, header=None)

    def _read_dr(self, dr_path, get_overlap_only=True):
        """
        (Private) Read the relation / domain / range text file.

        Inputs:
            dr_path: file path of the text file containing the
                domain / relation / range information
            get_overlap_only: True if only getting the DRR
                info that is available in the dataset
                (used to skip negative relations)

        Returns:
            pd_dr: dataframe containing the domain / range info
            all_relations_list: list containing all the relations
            entity_types: list containing all the unique entity types
        """
        pd_dr = pd.read_csv(dr_path, sep='\t')

        # get all the relations in the dataset working on
        relation_group = self.pd_data.groupby(self.PRED_STR)
        all_relations_list = list(relation_group.groups.keys())

        # find unique entity types
        entity_types = []
        for dr_tuple, _ in pd_dr.groupby([self.DOMAIN_STR, self.RANGE_STR]):
            entity_types.extend(list(dr_tuple))
        entity_types = list(set(entity_types))

        # get only the (domain / relation / range) data
        # that is available in the dataset
        if get_overlap_only:
            matching_relations_idx = pd_dr[self.RELATION_STR].isin(all_relations_list)
            pd_dr = pd_dr[matching_relations_idx].reset_index(drop=True)

        log.debug('(Relation / Domain /  Range) to process:\n%s', pd_dr)

        return pd_dr, all_relations_list, entity_types

    def _fill_entity_dic(self):
        """
        (Private) Complete the dictionary where the key is entity type
        and value is the numpy array containing all the entities belonging
        to that type.

        Returns:
            entity_dic: completed entity dictionary
        """
        entity_dic = dict.fromkeys(self.entity_types, np.array([]))

        # loop through each relation and fill in the
        # entity dictionary based on their domain / range type
        for relation in self.relations:
            log.debug('Processing relation: %s', relation)
            single_grr = self.pd_dr.loc[self.pd_dr[self.RELATION_STR] == relation]

            domain_type = single_grr[self.DOMAIN_STR].item()
            range_type = single_grr[self.RANGE_STR].item()

            domains, ranges = self._get_domain_range(relation)

            entity_dic[domain_type] = np.append(entity_dic[domain_type], domains)
            entity_dic[range_type] = np.append(entity_dic[range_type], ranges)

        for key, value in entity_dic.items():
            entity_dic[key] = np.unique(value)
            log.debug('Count of entity type \'%s\': %d', key, entity_dic[key].shape[0])

        return entity_dic

    def _get_domain_range(self, relation):
        """
        Given a relation, find all the unique domains and ranges.

        Inputs:
            relation: relation type

        Returns:
            all_unique_domains: all unique domains
            all_unique_ranges: all unique ranges
        """
        # get index which has specified relation type
        matching_relation_idx = self.pd_data[self.PRED_STR].isin([relation])

        # find unique domain & range for given relation type
        all_unique_domains = self.pd_data[matching_relation_idx][self.SUBJECT_STR].unique()
        all_unique_ranges = self.pd_data[matching_relation_idx][self.OBJECT_STR].unique()

        return all_unique_domains, all_unique_ranges
