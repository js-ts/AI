
from typing import Tuple, Union
from copy import deepcopy

from pdll.backend.executor import np, support_types
from .tensor import Tensor


__all__ = [
    'rand', 'randn', 'eye',
    'ones', 'ones_like',
    'zeros', 'zeros_like',
    'from_numpy', 'tensor',
]


def tensor(data: Union[support_types]) -> Tensor:
    return Tensor(deepcopy(data))


def rand(*shape: Tuple[int], requires_grad=False) -> Tensor:
    '''
    '''
    data = np.random.rand(*shape)
    return Tensor(data, requires_grad=requires_grad)


def randn(*shape: Tuple[int], requires_grad=False) -> Tensor:
    '''
    '''
    data = np.random.randn(*shape)
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape: Tuple[int], requires_grad=False) -> Tensor:
    '''
    '''
    data = np.zeros(shape)
    return Tensor(data, requires_grad=requires_grad)


def ones(*shape: Tuple[int], requires_grad=False) -> Tensor:
    '''
    '''
    data = np.ones(shape)
    return Tensor(data, requires_grad=requires_grad)


def zeros_like(v: Tensor, requires_grad=False) -> Tensor:
    '''
    '''
    data = np.zeros(v.shape, dtype=v.dtype)
    return Tensor(data, requires_grad=requires_grad)


def ones_like(v: Tensor, requires_grad=False) -> Tensor:
    '''
    '''
    data = np.ones(v.shape, dtype=v.dtype)
    return Tensor(data, requires_grad=requires_grad)


def from_numpy(data: Tensor, requires_grad=False) -> Tensor:
    '''
    '''
    return Tensor(data, requires_grad=requires_grad)


def eye(*shape, requires_grad=False):
    '''
    '''
    data = np.eye(shape)
    return Tensor(data, requires_grad=requires_grad)