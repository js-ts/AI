import numpy as np

from .tensor import Tensor
from .tensor import Dependency


class Function(object):

    @staticmethod
    def forward(*data):
        raise NotImplementedError('forward')

    @staticmethod
    def backward(*grad):
        raise NotImplementedError('backward')
