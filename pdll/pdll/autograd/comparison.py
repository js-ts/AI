from typing import Tuple, Iterable, Iterator, Union

from functools import reduce as REDUCE
from operator import mul as MUL

from pdll.backend import executor

from .function import Function
from .tensor import Tensor
from .backpropag import Leaf

from .utils import broadcast_reverse
from .utils import register
from .utils import to_tensor

import numbers


@register(Tensor)
def eq(tensor, other):
    '''equal
    '''
    if isinstance(other, (Tensor, numbers.Real)):
        other = to_tensor(other)
        data = tensor.storage == other.storage
        return Tensor(data)
    else:
        # return NotImplemented
        return id(tensor) == id(other)


def gt(tensor, other):
    '''tensor > other
    '''
    if isinstance(other, (Tensor, numbers.Real)):
        other = to_tensor(other)
        data = tensor.storage > other.storage
        return Tensor(data)
    else:
        return NotImplemented



@register(Tensor)
def __eq__(self, other):
    return eq(self, other)


@register(Tensor)
def __gt__(self, other):
    return gt(self, other)
