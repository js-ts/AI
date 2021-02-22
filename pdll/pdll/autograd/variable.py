from typing import Union

from pdll.backend import Tensor
from pdll.backend import np

from .backpropag import ExecuteEngine
from .backpropag import Leaf


class Variable(object):
    '''
    '''
    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        # assert isinstance(data, (Tensor, np.float32, np.float64)), f'{data} {type(data)}'
        assert isinstance(requires_grad, bool), ''

        if creator is None:
            creator = Leaf(self, requires_grad)
        self.creator = creator
        self.tensor = data
        self.grad = None
        self.shape = data.shape
        self.requires_grad = self.creator.requires_grad

    @property
    def data(self, ):
        return Variable(self.tensor)

    def numpy(self, ):
        '''numpy
        '''
        return self.tensor[...]


    def backward(self, grad=None):
        '''backward
        '''
        if grad is None:
            grad = Variable(np.ones_like(self.tensor))
        elif isinstance(grad, (int, float)):
            grad = Variable(np.array([grad]))
        elif isinstance(grad, Tensor):
            grad = Variable(grad)
        elif isinstance(grad, Variable):
            assert grad.shape == self.shape
        else:
            raise RuntimeError('-------')
        
        self._engine.backward_fn(self.creator, grad)

    def zero_grad(self, ):
        # assert self.grad is not None, ''
        if self.grad is not None:
            self.grad._data[...] = 0

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

    # def __setattr__(self, name, value):
        # assert name in self.__doc__, ''
        # object.__setattr__(self, name, value)
