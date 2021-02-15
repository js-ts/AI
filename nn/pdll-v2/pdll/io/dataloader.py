import numpy as np 
import math
import random

from .dataset import Dataset

class DataLoader(object):
    '''dataloader
    '''
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool=False, drop_last: bool=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.perms = np.arange(len(dataset))
        self.batch_idx = -1 
        if shuffle:
            self.perms = np.random.permutation(np.arange(len(self.dataset)))

    def __iter__(self):
        return self

    def __next__(self, ):
        '''
        '''
        self.batch_idx += 1
        if self.batch_idx >= math.ceil(len(self.dataset) // self.batch_size):
            self.batch_size = -1
            if self.shuffle:
                random.shuffle(self.perms)
            raise StopIteration

        batch = self.dataset[self.perms[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]]
        
        return batch


    def reset(self, ):
        '''
        '''
        self.batch_size = -1


    def next_batch(self, ):
        '''
        '''
        pass