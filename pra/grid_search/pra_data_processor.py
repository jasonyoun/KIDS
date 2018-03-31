import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv

class DataProcessor:


    def get_inverse(self, relation):
        if relation in self.known_inverses_dic:
            return self.known_inverses_dic[relation]
        else:
            return ''
    def __init__(self,data_path):
        self.known_inverses_dic = {}
        # self.known_inverses_dic['activates'] = 'represses'
        # self.known_inverses_dic['represses'] = 'activates'
        self.data_path = data_path

    def load(self):
        df = pd.read_csv(self.data_path+'/data.txt',sep='\t',encoding ='latin-1',header=0)
        return df

    def create_selected_relations_file(self, data ):
        self.data = data
        dic = {}
        relation_set = set()
        for i in range(len(data)):
            relation_set.add(data[i][1])
        self.relation_set = relation_set
        index_id = 0
        with open('selected_relations', 'a') as the_file:
            for r in relation_set:
                the_file.write(r+'\n')

    def create_relations_file(self ):
        with open('relations', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['relationName', 'humanFormat','populate','generalizations','domain','range','antisymmetric','mutexExceptions','knownNegatives','inverse','seedInstances','seedExtractionPatterns','nrOfValues','nrOfInverseValues','requiredForDomain','requiredForRange','editDate','author','description','freebaseID','coment' ])
            for r in self.relation_set:
                writer.writerow([r, '{{}}', 'true', '{}', 'object', 'object', 'true']+ ['(empty set)']*2+[self.get_inverse(r)]+['(empty set)']*2+['any']+['NO_THEO_VALUE']*8)

    def create_entity_set(self):
        entity_set = set()
        for i in range(len(self.data)):
            entity_set.add(self.data[i][0])
            entity_set.add(self.data[i][2])
        return entity_set

    def create_triplets_generalizations_file(self ):
        positives = self.data[self.data[:,3] == 1]
        entity_set = self.create_entity_set()
        with open('ecoli_generalizations.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Entity', 'Relation','Value','Iteration of Promotion','Probability','Source','Candidate Source'])
            for row in positives:
                writer.writerow([row[0], row[1], row[2], '', '1.0', '', ''])
            for k in entity_set:
                writer.writerow([k, 'generalizations', 'object', '', '0.2', '', ''])



if __name__ == "__main__":
    processor = DataProcessor('./')
    df = processor.load()
    processor.create_selected_relations_file(df.as_matrix())
    processor.create_relations_file()
    processor.create_triplets_generalizations_file()

