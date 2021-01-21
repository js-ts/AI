import numpy as np

from autograd.tensor import Tensor
from autograd.tensor import Dependency


class Function(object):

    @staticmethod
    def forward(*data):
        raise NotImplementedError('forward')

    @staticmethod
    def backward(*grad):
        raise NotImplementedError('backward')
