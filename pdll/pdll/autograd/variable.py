from typing import Union

from pdll.backend import Tensor
from pdll.backend import np, support_types

from .backpropag import ExecuteEngine
from .backpropag import Leaf


class Variable(object):
    '''
    '''
    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        assert isinstance(data, support_types), f'{data} {type(data)}'
        assert isinstance(requires_grad, bool), ''

        if creator is None:
            creator = Leaf(self, requires_grad)
        self.creator = creator
        self.tensor = data # storage
        self.grad = None
        self.requires_grad = self.creator.requires_grad

    @property
    def data(self, ):
        return Variable(self.tensor)

    def numpy(self, ):
        '''numpy
        '''
        return self.tensor[...]

    @property
    def shape(self, ):
        return self.tensor.shape

    def backward(self, grad=None):
        '''backward
        '''
        if grad is None:
            grad = self.__class__(np.ones_like(self.tensor))
        elif isinstance(grad, (int, float)):
            grad = self.__class__(np.array([grad]))
        elif isinstance(grad, Tensor):
            grad = self.__class__(grad)
        elif isinstance(grad, Variable):
            assert grad.shape == self.shape, ''
        else:
            raise RuntimeError('type(grad) dont support.')
        
        self._engine.backward_fn(self.creator, grad)

    def zero_grad(self, ):
        if self.grad is not None:
            self.grad.tensor[...] = 0

    def register_hook(self, name, hook):
        raise NotImplementedError

    def remove_hook(self, name):
        raise NotImplementedError 

    def __repr__(self, ):
        s = f'Variable({self.tensor}'
        if self.requires_grad:
            s += f', requires_grad={self.requires_grad})'
        else:
            s += ')'
        return s
