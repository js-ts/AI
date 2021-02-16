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
        self.indices = np.arange(len(dataset))
        self._iterator = None

    def __iter__(self):
        '''
        '''
        if self._iterator is None:
            self._iterator = _BaseDataLoaderIter(self)
        else:
            self._iterator.reset()

        return self._iterator

    def __len__(self, ):
        return len(self.dataset)


class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.indices = loader.indices
        self.batch_size = loader.batch_size
        self.drop_last = loader.drop_last
        self.batch_idx = -1
        self.shuffle = loader.shuffle

        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self, ):
        return self

    def __next__(self, ):
        '''
        '''
        self.batch_idx += 1
        if self.batch_idx >= len(self.indices) // self.batch_size:
            raise StopIteration

        _idx = self.indices[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]
        batch = [self.loader.dataset[i] for i in _idx]

        if not isinstance(batch[0], tuple):
            return batch
        else:
            batch = list(zip(*batch))
            return batch


    def reset(self, ):
        self.batch_idx = -1
        if self.shuffle:
            random.shuffle(self.indices)
    