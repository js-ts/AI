
from typing import Iterable

from ..nn import Parameter


class Optimizer(object):
    def __init__(self, params: Iterable[Parameter], lr: float):
        self.lr = lr
        self.params = params

    def step(self, ):
        raise NotImplementedError

    def zero_grad(self, ):
        for p in self.params:
            p.zero_grad()


class SGD(Optimizer):
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self, ):
        for p in self.params:
            p.data = p.data - p.grad * self.lr

    def zero_grad(self, ):
        for p in self.params:
            p.zero_grad()