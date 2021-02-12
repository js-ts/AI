
from .dataset import Dataset

class DataLoader(object):
    '''dataloader
    '''
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self, ):
        raise NotImplementedError