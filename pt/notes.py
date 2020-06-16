import random
import numpy as np
import torch

import matplotlib.pyplot as plt


def set_random(seed_val=0):
    '''
    '''
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
