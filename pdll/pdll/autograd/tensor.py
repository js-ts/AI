from typing import Union

from pdll.backend import np, support_types

from .backpropag import ExecuteEngine
from .backpropag import Leaf


class Tensor(object):
    '''
    '''
    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        assert isinstance(data, support_types), f'{data} {type(data)}'
        assert isinstance(requires_grad, bool), ''

        if creator is None:
            creator = Leaf(self, requires_grad)
        self.creator = creator

        self.storage = data # storage
        self._grad = None

    @property
    def data(self, ):
        return Tensor(self.storage)

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor), ''
        self = value

    def numpy(self, ):
        '''cpu numpy
        '''
        return self.storage[...]

    @property
    def grad(self, ):
        return self._grad
    
    @grad.setter
    def grad(self, value):
        '''set grad
        '''
        assert isinstance(value, Tensor), ''
        self._grad = value

    @property
    def shape(self, ):
        return self.storage.shape

    @property
    def dtype(self, ):
        return self.storage.dtype
        
    @property
    def requires_grad(self, ):
        return self.creator.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        assert isinstance(value, bool), ''
        self.creator.requires_grad = value
    
    @property
    def device():
        return 'cpu'

    def backward(self, grad=None):
        '''backward
        '''
        if grad is None:
            grad = self.__class__(np.ones_like(self.storage))
        elif isinstance(grad, (int, float)):
            grad = self.__class__(np.array([grad]))
        elif isinstance(grad, support_types):
            grad = self.__class__(grad)
        elif isinstance(grad, self.__class__):
            assert grad.shape == self.shape, ''
        else:
            raise RuntimeError('type(grad) dont support.')
        
        self._engine.backward_fn(self.creator, grad)

    def zero_grad(self, ):
        if self.grad is not None:
            self.grad.storage[...] = 0

    def register_hook(self, name, hook):
        raise NotImplementedError

    def remove_hook(self, name):
        raise NotImplementedError 

    def __repr__(self, ):
        s = f'Tensor({self.storage}'
        if self.requires_grad:
            s += f', requires_grad={self.requires_grad})'
        else:
            s += ')'
        return s
