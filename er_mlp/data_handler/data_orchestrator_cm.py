import numpy as np
import random
import math
import pandas as pd

def generate_neg(subject,type_dic,domain_range_dic,set_of_neg_objects,set_of_pos_objects, _range, entity_dic, use_range=True, use_neg=True):
    if use_range:
        print('should not be in here')
        if len(set_of_neg_objects)>0 and use_neg:
            _object = random.sample(set_of_neg_objects, 1)[0]
            set_of_neg_objects.remove(_object)
        else:
            _object = random.sample(_range,1)[0]
            while _object in set_of_pos_objects:
                _object = random.sample(_range,1)[0]
    else:
        _object = random.randint(0, len(entity_dic)-1)
    return _object

def generate_freebase_neg( entity_dic):
    _object = random.randint(0, len(entity_dic)-1)
    return _object
def clean_row(row):
    for i in range(np.shape(row)[0]):
        if isinstance(row[i], str):
            row[i] = row[i].strip()

def create_type_subsets_dic(data_array,entity_dic):
    type_dic = {}
    subsets_dic = {}
    for row in data_array:
        clean_row(row)
        if row[1] not in subsets_dic:
            subsets_dic[row[1]] = set()
        subsets_dic[row[1]].add(entity_dic[row[2]])
        type_dic[entity_dic[row[2]]] = row[1]
    return type_dic,subsets_dic

def create_predicate_to_entity_to_triplets_set(data_array, positive=True ):
    isPositive=-1
    if positive:
        isPositive=1
    predicate_to_entity_to_triplets = {}
    for row in data_array:
        clean_row(row)
        if row[3] == isPositive:
            if row[1] not in predicate_to_entity_to_triplets:
                predicate_to_entity_to_triplets[row[1]] = {}
            if row[0] not in predicate_to_entity_to_triplets[row[1]]:
                predicate_to_entity_to_triplets[row[1]][row[0]] = set()
            predicate_to_entity_to_triplets[row[1]][row[0]].add(row[2])
    return predicate_to_entity_to_triplets

def create_predicate_to_entity_to_subject_set(data_array, positive=True ):
    isPositive=-1
    if positive:
        isPositive=1
    predicate_to_entity_to_subjects = {}
    for row in data_array:
        clean_row(row)
        if row[3] == isPositive:
            if row[1] not in predicate_to_entity_to_subjects:
                predicate_to_entity_to_subjects[row[1]] = {}
            if row[2] not in predicate_to_entity_to_subjects[row[1]]:
                predicate_to_entity_to_subjects[row[1]][row[2]] = set()
            predicate_to_entity_to_subjects[row[1]][row[2]].add(row[0])
    return predicate_to_entity_to_subjects


class DataOrchestrator:

    def get_type_subsets_dic(self):

        return self.type_dic,self.subsets_dic

    def __init__(self, data_set, data_path,predicate_dic,entity_dic, corruption_size=20, shuffle=True, use_range=True, use_neg=True,is_freebase=False):
        print(predicate_dic)
        self.shuffle=shuffle
        self.data_index = 0
        self.data_set = data_set
        self.data_path = data_path
        self.pos_data_set =  data_set[data_set[:,3] == 1 ]
        self.entity_dic = entity_dic
        self.use_range = use_range
        self.use_neg = use_neg
        self.is_freebase = is_freebase

        self.corruption_size = corruption_size
        if self.shuffle:
            self.shuffle_data()
        if not self.is_freebase:
            print('create subsets dic')
            print('')
            my_file = "entity_full_names.txt"
            df = pd.read_csv(self.data_path+'/'+my_file,sep=':',encoding ='latin-1',header=None)
            data_array = df.as_matrix()
            self.type_dic,self.subsets_dic = create_type_subsets_dic(data_array,entity_dic)

            my_file = "domain_range.txt"
            df = pd.read_csv(self.data_path+'/'+my_file,sep='\t',encoding ='latin-1',header=None)
            data_array = df.as_matrix()
            self.domain_range_dic = {}
            for row in data_array:
                self.domain_range_dic[predicate_dic[row[0].strip()]] = (row[1].strip(), row[2].strip())

            self.predicate_to_entity_to_neg_triplets = create_predicate_to_entity_to_triplets_set(self.data_set, positive=False )
            self.predicate_to_entity_to_pos_triplets = create_predicate_to_entity_to_triplets_set(self.data_set, positive=True )

            self.predicate_to_entity_to_neg_subjects = create_predicate_to_entity_to_subject_set(self.data_set, positive=False )
            self.predicate_to_entity_to_pos_subjects = create_predicate_to_entity_to_subject_set(self.data_set, positive=True )

            self.predicate_to_entity_to_neg_triplets_per_epoch = self.predicate_to_entity_to_neg_triplets.copy()
            self.predicate_to_entity_to_pos_triplets_per_epoch = self.predicate_to_entity_to_pos_triplets.copy()

            self.predicate_to_entity_to_neg_subjects_per_epoch = self.predicate_to_entity_to_neg_subjects.copy()
            self.predicate_to_entity_to_pos_subjects_per_epoch = self.predicate_to_entity_to_pos_subjects.copy()



    def shuffle_data(self):
        np.random.shuffle(self.pos_data_set)


    def reset_data_index(self):
        self.data_index = 0
        if not self.is_freebase:
            self.predicate_to_entity_to_neg_triplets_per_epoch = self.predicate_to_entity_to_neg_triplets.copy()
            self.predicate_to_entity_to_pos_triplets_per_epoch = self.predicate_to_entity_to_pos_triplets.copy()
            self.predicate_to_entity_to_neg_subjects_per_epoch = self.predicate_to_entity_to_neg_subjects.copy()
            self.predicate_to_entity_to_pos_subjects_per_epoch = self.predicate_to_entity_to_pos_subjects.copy()
        if self.shuffle:
            self.shuffle_data()

    def get_next_training_batch(self,batch_size,flip):
        positive_triplets = self.pos_data_set[self.data_index:self.data_index + batch_size]
        self.data_index+=batch_size
        batch = []
        for i in range(np.shape(positive_triplets)[0]):
            
            subject = positive_triplets[i][0]
            predicate = positive_triplets[i][1]
            correct_object = positive_triplets[i][2]
            if not self.is_freebase:
                if flip:
                    if predicate in self.predicate_to_entity_to_neg_triplets_per_epoch and subject in self.predicate_to_entity_to_neg_triplets_per_epoch[predicate]:
                        set_of_neg_objects = self.predicate_to_entity_to_neg_triplets_per_epoch[predicate][subject]
                    else:
                        set_of_neg_objects = set()

                    if predicate in self.predicate_to_entity_to_pos_triplets_per_epoch and subject in self.predicate_to_entity_to_pos_triplets_per_epoch[predicate]:
                        set_of_pos_objects = self.predicate_to_entity_to_pos_triplets_per_epoch[predicate][subject]
                    else:
                        set_of_pos_objects = set()
                else:
                    if predicate in self.predicate_to_entity_to_neg_subjects_per_epoch and correct_object in self.predicate_to_entity_to_neg_subjects_per_epoch[predicate]:
                        set_of_neg_objects = self.predicate_to_entity_to_neg_subjects_per_epoch[predicate][correct_object]
                    else:
                        set_of_neg_objects = set()

                    if predicate in self.predicate_to_entity_to_pos_subjects_per_epoch and correct_object in self.predicate_to_entity_to_pos_subjects_per_epoch[predicate]:
                        set_of_pos_objects = self.predicate_to_entity_to_pos_subjects_per_epoch[predicate][correct_object]
                    else:
                        set_of_pos_objects = set()
                if flip:
                    _range = self.subsets_dic[self.domain_range_dic[predicate][1]]
                else:
                    _range = self.subsets_dic[self.domain_range_dic[predicate][0]]
            for j in range(self.corruption_size):
                if not self.is_freebase:
                    corrupted = generate_neg(subject,self.type_dic,self.domain_range_dic,set_of_neg_objects,set_of_pos_objects,_range,self.entity_dic, self.use_range,self.use_neg)
                else:
                    corrupted = generate_freebase_neg(self.entity_dic)
                batch.append([subject, predicate, correct_object, corrupted])
        return batch






            