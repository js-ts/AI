import numpy as np 
from typing import Union, List

from .variable import Variable
from .tensor import Tensor


def to_tensor(data, dtype=np.float64):
    if isinstance(data, Tensor):
        return data.astype(dtype)
    else:
        return np.array(data).astype(dtype)


def to_variable(data):
    '''make sure data is variable
    '''
    if isinstance(data, (int, float)):
        data = to_tensor(data)
        return Variable(data)

    elif isinstance(data, (list, tuple)):
        data = to_tensor(data)
        return Variable(data)

    elif isinstance(data, Tensor):
        return Variable(data)
        
    elif isinstance(data, Variable):
        return data
    
    else:
        raise RuntimeError('not support data type.')