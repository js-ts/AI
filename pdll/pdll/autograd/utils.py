from typing import Union, List, Tuple, Iterable
import functools

from ..backend import np, support_types
from .tensor import Tensor

__all__ = [
    'to_tensor',
    'broadcast_reverse',
    'register',
]


def _to_numpy(data, dtype=np.float32):
    return np.array(data).astype(dtype)

def to_tensor(data):
    '''make sure data is Tensor
    '''
    if isinstance(data, (int, float)):
        data = _to_numpy(data)
        return Tensor(data)

    elif isinstance(data, (list, tuple)):
        data = _to_numpy(data)
        return Tensor(data)
        
    elif isinstance(data, Tensor):
        return data

    else:
        raise RuntimeError('not support data type.')


def broadcast_reverse(grad, shape: Iterable[int]): 
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


def register(cls=Tensor):
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