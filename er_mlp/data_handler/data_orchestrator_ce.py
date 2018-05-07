import numpy as np
import random
import math
import pandas as pd


class DataOrchestrator:

    def __init__(self, data_set,shuffle=False):
        self.shuffle=shuffle
        self.data_index = 0
        self.data_set = data_set
        if self.shuffle:
            self.shuffle_data()



    def shuffle_data(self):
        np.random.shuffle(self.data_set)


    def reset_data_index(self):
        self.data_index = 0
        if self.shuffle:
            self.shuffle_data()


    def get_next_training_batch(self,batch_size):
        batch = self.data_set[self.data_index:self.data_index + batch_size]
        self.data_index+=batch_size
        return batch





            