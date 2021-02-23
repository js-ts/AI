
from ..backend import np 
from ..backend import Tensor
from .variable import Variable

from typing import Tuple

__all__ = [
    'rand', 'randn', 'eye',
    'ones', 'ones_like',
    'zeros', 'zeros_like',
    'from_numpy',
]

def rand(*shape: Tuple[int], requires_grad=False) -> Variable:
    '''
    '''
    data = np.random.rand(*shape)
    return Variable(data, requires_grad=requires_grad)


def randn(*shape: Tuple[int], requires_grad=False) -> Variable:
    '''
    '''
    data = np.random.randn(*shape)
    return Variable(data, requires_grad=requires_grad)


def zeros(*shape: Tuple[int], requires_grad=False) -> Variable:
    '''
    '''
    data = np.zeros(shape)
    return Variable(data, requires_grad=requires_grad)


def ones(*shape: Tuple[int], requires_grad=False) -> Variable:
    '''
    '''
    data = np.ones(shape)
    return Variable(data, requires_grad=requires_grad)


def zeros_like(v: Variable, requires_grad=False) -> Variable:
    '''
    '''
    data = np.zeros(v.shape)
    return Variable(data, requires_grad=requires_grad)


def ones_like(v: Variable, requires_grad=False) -> Variable:
    '''
    '''
    data = np.ones(v.shape)
    return Variable(data, requires_grad=requires_grad)


def from_numpy(data: Tensor, requires_grad=False) -> Variable:
    '''
    '''
    return Variable(data, requires_grad=requires_grad)


def eye(*shape, requires_grad=False):
    '''
    '''
    data = np.eye(shape)
    return Variable(data, requires_grad=requires_grad)