
class Dataset(object):
    '''dataset
    '''
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self, ):
        raise NotImplementedError

    



class MNIST(Dataset):

    def __init__(self, root):
        '''
        '''
        self.root = root
        
    def len(self, ):
        return 0
        