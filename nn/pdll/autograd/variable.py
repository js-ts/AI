import numpy as np

from .tensor import Tensor
from .engine import ExecuteEngine

from .utils import to_tensor, to_variable
from .operator import Leaf, Add


class Variable(object):
    '''
    '''
    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        if creator is None:
            creator = Leaf(self, requires_grad)
        self.data = data
        self.creator = creator
        self.shape = self.data.shape
        self.requires_grad = self.creator.requires_grad
        
        self.grad = None
        if isinstance(creator, Leaf) and requires_grad:
            self.grad = np.zeros_like(data)

    def backward(self, grad=1.):
        if not isinstance(grad, Variable):
            grad = to_variable(grad)
        self._engine._backward_fn(self.creator, grad)

    def zero_grad(self, ):
        self.grad[...] = 0

    def register_hook(self, name, hook):
        raise NotImplementedError

    def remove_hook(self, name):
        raise NotImplementedError 


    # basic op
    def add(self, other):
        other = to_variable(other)
        return Add()(self, other)[0]


    # magic method
    def __add__(self, other):
        return self.add(other)
