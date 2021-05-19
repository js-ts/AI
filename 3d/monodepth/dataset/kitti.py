import torch
import torch.utils.data as data


import numpy as np 


class KITIIDataset(data.Dataset):
    def __init__(self, filename):
        super().__init__()

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.im_size = (1242, 375)
        self.side_map = {'2': 2, '3': 3, 'l': 2, 'r': 3}

        self.lines = []
        
    def __len__(self, ):
        pass

    def __getitem__(self, index):
        pass
