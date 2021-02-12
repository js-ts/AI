
class Dataset(object):
    '''dataset
    '''
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self, ):
        raise NotImplementedError