import numpy as np

from autograd.tensor import Tensor
from autograd.tensor import Dependency

def tanh(t: Tensor) -> Tensor:
    data = np.tanh(t.data)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            '''tanh(x)
            '''
            return grad * (1 - data * data)
            
        depends_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)


class Function(object):

    @staticmethod
    def forward(*data):
        raise NotImplementedError('forward')

    @staticmethod
    def backward(*grad):
        raise NotImplementedError('backward')
