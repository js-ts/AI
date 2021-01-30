
from typing import Iterable
from ..nn import Parameter

class Optimizer(object):
    def __init__(self, params: Iterable[Parameter], lr: float, lr_scheduler=None):
        self.lr = lr
        self.params = params
        self.lr_scheduler = lr_scheduler

    def step(self, ):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)
    
    def step(self, ):
        for p in self.params:
            p.data -= self.lr * p.data