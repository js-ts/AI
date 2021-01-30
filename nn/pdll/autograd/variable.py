import numpy as np

from .tensor import Tensor
from .engine import ExecuteEngine



class Variable(object):
    '''
    '''
    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        from .operator import Leaf
        
        if creator is None:
            creator = Leaf(self, requires_grad)
        self.data = data
        self.creator = creator
        self.shape = self.data.shape
        self.requires_grad = self.creator.requires_grad
        
        self.grad = None
        if isinstance(creator, Leaf) and requires_grad:
            self.grad = np.zeros_like(data)

    def backward(self, grad=None):
        '''backward
        '''
        if grad == None:
            grad = to_variable(np.ones_like(self.data))
        elif isinstance(grad, (int, float)):
            grad = to_variable(grad)
        else:
            assert isinstance(grad, Variable) and grad.shape == self.shape, ''

        self._engine.backward_fn(self.creator, grad.data)

    def zero_grad(self, ):
        self.grad[...] = 0

    def register_hook(self, name, hook):
        raise NotImplementedError

    def remove_hook(self, name):
        raise NotImplementedError 

    def __repr__(self, ):
        return f'Variable(data={self.data}, requires_grad={self.requires_grad})'


    # basic-op
    def add(self, other):
        other = to_variable(other)
        return Add()(self, other)[0]

    def sub(self, other):
        other = to_variable(other)
        return Sub()(self, other)[0]

    def neg(self, ):
        return Neg()(self)[0]

    def mul(self, other):
        other = to_variable(other)
        return Mul()(self, other)[0]

    def div(self, other):
        other = to_variable(other)
        return Div()(self, other)[0]

    def matmul(self, other):
        # other = to_variable(other)
        return Matmul()(self, other)[0]

    def pow(self, n):
        return Pow(n)(self)[0]

    def sqrt(self, ):
        return Pow(1/2.)(self)[0]
        
    def exp(self, ):
        return Exp()(self)[0]

    def rpow(self, a):
        return RPow(a)(self)[0]

    def sum(self, axis=None, keepdims=False):
        return Sum(axis, keepdims)(self)[0]

    def mean(self, axis=None, keepdims=False):
        return Mean(axis, keepdims)(self)[0]

    def var(self, axis=None, keepdims=False):
        return ((self - self.mean(axis, True)) ** 2).mean(axis, keepdims)

    def reshape(self, *shape):
        return Reshape(*shape)(self)[0]

    def transpose(self, *dims):
        return Transpose(*dims)(self)[0]


    # magic-method
    def __add__(self, other):
        '''self + other
        '''
        return self.add(other)

    def __radd__(self, other):
        '''other + self
        '''
        other = to_variable(other)
        return other.add(self)

    def __sub__(self, other):
        return self.sub(other)
    
    def __rsub__(self, other):
        other = to_variable(other)
        return other.sub(self)

    def __neg__(self, ):
        return self.neg()

    def __mul__(self, other):
        return self.mul(other)
    
    def __rmul__(self, other):
        other = to_variable(other)
        return other.mul(self)

    def __div__(self, other):
        return self.div(other)
    __truediv__ = __div__

    def __rdiv__(self, other):
        other = to_variable(other)
        return other.div(self)
    __rtruediv__ = __rdiv__

    def __matmul__(self, other):
        return self.matmul(other)

    def __pow__(self, n):
        return self.pow(n)

    def __rpow__(self, a):
        return self.rpow(a)

    def __getitem__(self, idx):
        return GetItem(idx)(self)[0]


from .utils import to_variable, to_tensor
from .operator import Leaf
from .operator import Add, Sub, Mul, Div
from .operator import Neg, Pow, Exp, RPow
from .operator import Matmul
from .operator import Sum, Mean
from .operator import Reshape, Transpose
from .operator import GetItem