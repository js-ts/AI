import numpy as np 

from .tensor import Tensor

class Loss:
    def loss(self, pred: Tensor, label: Tensor) -> Tensor:
        raise NotImplementedError

    def grad(self, pred: Tensor, label: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):

    def loss(self, pred: Tensor, label: Tensor) -> Tensor:
        return np.sum((pred - label) ** 2)

    def grad(self, pred: Tensor, label: Tensor) -> Tensor:
        return 2 * (pred - label)
    
    