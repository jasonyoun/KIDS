import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle
import random
import scipy.io as spio
import csv

def create_type_subsets_dic(data_array):
    type_dic = {}
    subsets_dic = {}
    for row in data_array:
        clean_row(row)
        if row[1] not in subsets_dic:
            subsets_dic[row[1]] = set()
        subsets_dic[row[1]].add(row[2])
        type_dic[row[2]] = row[1]
    return type_dic,subsets_dic

def clean_row(row):

    for i in range(np.shape(row)[0]):
        if isinstance(row[i], str):
            row[i] = row[i].strip()

class DataProcessor:

    def __init__(self,data_path, data_file,freebase=False):
        self.freebase = freebase
        self.data_path = data_path
        print('create subsets dic')
        print('')
        if not self.freebase:
            my_file = "entity_full_names.txt"
            df = pd.read_csv(self.data_path+'/'+my_file,sep=':',encoding ='latin-1',header=None)
            data_array = df.as_matrix()
            self.type_dic,self.subsets_dic = create_type_subsets_dic(data_array)

            my_file = "domain_range.txt"
            df = pd.read_csv(self.data_path+'/'+my_file,sep='\t',encoding ='latin-1',header=None)
            data_array = df.as_matrix()
            self.domain_range_dic = {}
            for row in data_array:
                self.domain_range_dic[row[0].strip()] = (row[1].strip(), row[2].strip())

            self.no_negatives = set()
            self.no_negatives.add('has')
            self.no_negatives.add('is')
            self.no_negatives.add('is#SPACE#involved#SPACE#in')
            self.no_negatives.add('upregulated#SPACE#by#SPACE#antibiotic')
            self.no_negatives.add('targeted#SPACE#by')



    def load(self, data_file):
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

    def create_sets(self ):
        entities_file = self.data_path+"/entities.txt"
        relations_file = self.data_path+"/relations.txt"
        entity_set = set()
        relation_set = set()
        with open(entities_file) as f:
            for line in f:
                entity_set.add(line.strip())
        with open(relations_file) as f:
            for line in f:
                relation_set.add(line.strip())
        self.relation_set = relation_set
        self.entity_set = entity_set


    def create_relations_file(self ):
        with open('relations', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['relationName', 'humanFormat','populate','generalizations','domain','range','antisymmetric','mutexExceptions','knownNegatives','inverse','seedInstances','seedExtractionPatterns','nrOfValues','nrOfInverseValues','requiredForDomain','requiredForRange','editDate','author','description','freebaseID','coment' ])
            for r in self.relation_set:
                if not self.freebase:
                    writer.writerow(['concept:'+r, '{{}}', 'true', '{"object"}', self.domain_range_dic[r][0], self.domain_range_dic[r][1], 'true']+ ['concept:'+self.domain_range_dic[r][0],'concept:'+self.domain_range_dic[r][1]]+['']+['(empty set)']*2+['any']+['NO_THEO_VALUE']*8)
                else:
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
                # if not positive:
                #     if row[1] in self.no_negatives:
                #         continue
                # print(row[0])
                # print(row[1])
                
            if not self.freebase:
                for row in positives:
                    writer.writerow(['concept:'+self.type_dic[row[0]]+':'+row[0], 'concept:'+row[1], 'concept:'+self.type_dic[row[2]]+':'+row[2], '', '1.0', '', ''])
                for k in self.entity_set:
                    writer.writerow(['concept:'+k, 'generalizations', 'concept:'+self.type_dic[k], '', '1.0', '', ''])
                for k,v in self.subsets_dic.items():
                    writer.writerow(['concept:'+k, 'generalizations', 'concept:object', '', '1.0', '', ''])
            else:
                for row in positives:
                    writer.writerow([row[0], row[1], row[2], '', '1.0', '', ''])
                for k in self.entity_set:
                    writer.writerow([k, 'generalizations', 'object', '', '1.0', '', ''])



if __name__ == "__main__":
    freebase=False
    if len(sys.argv)>3:
        freebase=True
    processor = DataProcessor(sys.argv[1],sys.argv[2],freebase=freebase )
    test_df = processor.load('test.txt')
    processor.create_selected_relations_file(test_df.as_matrix())
    processor.create_sets()
    processor.create_relations_file()
    train_df = processor.load(sys.argv[2] )
    shape = np.shape(train_df.as_matrix())
    processor.create_triplets_generalizations_file(train_df.as_matrix())
    if not freebase:
        if (shape[1]==4):
            processor.create_triplets_generalizations_file(train_df.as_matrix(),positive=False)

