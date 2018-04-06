import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv

class DataProcessor:

    # def get_inverse(self, relation):
    #     if relation in self.known_inverses_dic:
    #         return self.known_inverses_dic[relation]
    #     else:
    #         return ''
    def __init__(self,data_path, data_file="/data.txt"):
        # self.known_inverses_dic = {}
        # self.known_inverses_dic['activates'] = 'represses'
        # self.known_inverses_dic['represses'] = 'activates'
        self.data_path = data_path

    def load(self, data_file='data.txt'):
        df = pd.read_csv(self.data_path+'/'+data_file,sep='\t',encoding ='latin-1',header=None)
        return df

    def create_selected_relations_file(self, data ):
        dic = {}
        selected_relation_set = set()
        for i in range(len(data)):
            selected_relation_set.add(data[i][1])
        self.selected_relation_set = selected_relation_set
        index_id = 0
        with open('selected_relations', 'a') as the_file:
            for r in selected_relation_set:
                the_file.write(r+'\n')

    def create_sets(self, data ):
        dic = {}
        relation_set = set()
        for i in range(len(data)):
            relation_set.add(data[i][1])
        self.relation_set = relation_set
        entity_set = set()
        for i in range(len(data)):
            entity_set.add(data[i][0])
            entity_set.add(data[i][2])
        self.entity_set = entity_set


    def create_relations_file(self ):
        with open('relations', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['relationName', 'humanFormat','populate','generalizations','domain','range','antisymmetric','mutexExceptions','knownNegatives','inverse','seedInstances','seedExtractionPatterns','nrOfValues','nrOfInverseValues','requiredForDomain','requiredForRange','editDate','author','description','freebaseID','coment' ])
            for r in self.selected_relation_set:
                writer.writerow([r, '{{}}', 'true', '{}', 'object', 'object', 'true']+ ['(empty set)']*2+['']+['(empty set)']*2+['any']+['NO_THEO_VALUE']*8)


    def create_triplets_generalizations_file(self, data, positive=True ):
        filename = 'ecoli_generalizations.csv'
        if np.shape(data)[1]!=4:
            #then it only contains postive triplets without the indicator
            new_col  = np.tile('1', data.shape[0])[None].T 
            data = np.concatenate((data, new_col), 1)

        positives = data[data[:,3] == '1']
        if np.shape(positives)[0]==0:
            positives = data[data[:,3] == 1 ]

        if not positive:
            positives = data[data[:,3] == '-1' ]
            if np.shape(positives)[0]==0:
                positives = data[data[:,3] == -1  ]
            if np.shape(positives)[0]==0:
                positives = data[data[:,3] == '0' ]
            if np.shape(positives)[0]==0:
                positives = data[data[:,3] == 0 ]
            filename = 'ecoli_generalizations_neg.csv'
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Entity', 'Relation','Value','Iteration of Promotion','Probability','Source','Candidate Source'])
            for row in positives:
                writer.writerow([row[0], row[1], row[2], '', '1.0', '', ''])
            for k in self.entity_set:
                writer.writerow([k, 'generalizations', 'object', '', '1.0', '', ''])



if __name__ == "__main__":
    if len(sys.argv)==3:
        processor = DataProcessor(sys.argv[1],sys.argv[2] )
    else:
        processor = DataProcessor(sys.argv[1])
    df = processor.load()
    test_df = processor.load('test.txt')
    processor.create_selected_relations_file(test_df.as_matrix())
    processor.create_sets(df.as_matrix())
    processor.create_relations_file()
    train_df = processor.load('train.txt')
    shape = np.shape(train_df.as_matrix())
    processor.create_triplets_generalizations_file(train_df.as_matrix())
    if (shape[1]==4):
        processor.create_triplets_generalizations_file(train_df.as_matrix(),positive=False)

