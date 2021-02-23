from typing import Union, List, Tuple, Iterable
import functools

from ..backend import np
from ..backend import Tensor
from .variable import Variable


def to_tensor(data, dtype=np.float32):
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


def broadcast_reverse(grad: Tensor, shape: Iterable[int]) -> Tensor: 
    '''reverse grad to shape
    '''
    _extdims = grad.ndim - len(shape)
    if _extdims:
        grad = grad.sum(axis=tuple(range(_extdims)))

    assert len(grad.shape) == len(shape), ''

    _axis = (i for i, d in enumerate(shape) if d == 1)
    if _axis:
        grad = grad.sum(axis=tuple(_axis), keepdims=True)

    assert grad.shape == shape, ''
    
    return grad


def register(cls=Variable):
    '''register
    '''
    def decorator(func):
        @functools.wraps(func)
        def wraps(*args, **kwargs):
            return func(*args, **kwargs)
        assert func.__name__ not in cls.__dict__, f'{func.__name__} already in ({cls.__name__})'
        setattr(cls, func.__name__, wraps)
        return wraps
    return decorator