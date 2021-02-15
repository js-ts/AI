
from typing import Iterable

from ..nn import Parameter


class _Optimizer(object):
    def __init__(self, params: Iterable[Parameter], lr: float):
        self.lr = lr
        self.params = list(params)
        assert isinstance(self.params, (tuple, list)), ''

    def step(self, ):
        raise NotImplementedError

    def zero_grad(self, ):
        for p in self.params:
            p.zero_grad()


class SGD(_Optimizer):
    '''https://pytorch.org/docs/stable/optim.html?highlight=sgd#torch.optim.SGD
    '''
    def __init__(self, params, lr: float, momentum: float=0.9, nesterov: bool=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = [0. for _ in self.params]

    def step(self, ):        
        # _params should not is generator
        # after loop, none
        for i, p in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] + (self.lr if self.velocity else 1.0) * p.grad
            p.data[...] = p.data - (1.0 if self.velocity else self.lr) * self.velocity[i]
            # if not self.nesterov:
            #     self.velocity[i] = self.momentum * self.velocity[i] + p.grad
            #     p.data[...] = p.data - self.lr * self.velocity[i]
            # else:
            #     self.velocity[i] = self.momentum * self.velocity[i] + self.lr * p.grad
            #     p.data[...] = p.data - self.velocity[i]

    def state_dict(self, ):
        '''
        '''
        raise NotImplementedError