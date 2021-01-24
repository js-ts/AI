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



def _sigmoid(arr: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-arr))

def _sigmoid_prime(arr: np.ndarray) -> np.ndarray:
    return _sigmoid(arr) * (1 - _sigmoid(arr))


class sigmoid(Function):

    def forward(self, t: Tensor) -> Tensor:
        self.t = t
        data = _sigmoid(t.data)

        return Tensor(data, Dependency([t], self.backward))
    
    def backward(self, grad: Tensor) -> Tensor:
        t = self.t
        return grad.data * _sigmoid_prime(t.data)