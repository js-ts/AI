from typing import Tuple, Iterable, Iterator

from .tensor import Tensor
from .function import Function

import numpy as np
from functools import reduce
import operator


def broadcast_reverse(grad: Tensor, shape: Tuple[int, ...]) -> Tensor: 
    '''reverse grad to shape
    '''
    _extdims = grad.ndim - len(shape)
    for _ in range(_extdims):
        grad = grad.sum(axis=0)
    assert len(grad.shape) == len(shape), ''

    for i, d in enumerate(shape):
        if d == 1:
            grad = grad.sum(axis=i, keepdims=True)
    assert grad.shape == shape, ''
    
    return grad


class Add(Function):
    '''a + b
    add broadcast
    [1, 3] + [2, 4, 3] -> [2, 4, 3]
    '''
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        c = a + b
        self.a_shape = a.shape
        self.b_shape = b.shape
        self.c_shape = c.shape
        return c

    def backward(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        assert self.c_shape == grad.shape, 'add' 
        a_grad = broadcast_reverse(grad, self.a_shape)
        b_grad = broadcast_reverse(grad, self.b_shape)
        return a_grad, b_grad


class Sub(Function):
    '''a - b
    '''
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        c = a - b
        self.a_shape = a.shape
        self.b_shape = b.shape
        self.c_shape = c.shape
        return c

    def backward(self, grad):
        assert grad.shape == self.c_shape, 'sub'
        a_grad = broadcast_reverse( grad, self.a_shape)
        b_grad = broadcast_reverse(-grad, self.b_shape)
        return a_grad, b_grad 


class Neg(Function):
    '''-t 
    '''
    def forward(self, t: Tensor) -> Tensor:
        return -t 
    
    def backward(self, grad: Tensor) -> Tensor:
        return -grad


class Mul(Function):
    '''a * b
    '''
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        c = a * b
        self.a = a
        self.b = b
        self.c_shape = c.shape
        return c
    
    def backward(self, grad):
        assert self.c_shape == grad.shape, 'mul'
        a_grad = broadcast_reverse(grad * self.b, self.a.shape)
        b_grad = broadcast_reverse(grad * self.a, self.b.shape)
        return a_grad, b_grad


class Div(Function):
    '''a / b
    '''
    def forward(self, a: Tensor, b: Tensor, eps: float=1e-10) -> Tensor:
        # np.testing.assert_almost_equal(b, 0)
        c = a / b
        self.a = a
        self.b = b
        self.c_shape = c.shape
        self.eps = eps
        return c
    
    def backward(self, grad: Tensor):
        assert grad.shape == self.c_shape
        a_grad = grad / self.b
        b_grad = -grad * self.a / (self.b ** 2 + 1e-10)
        a_grad = broadcast_reverse(a_grad, self.a.shape)
        b_grad = broadcast_reverse(b_grad, self.b.shape)
        return a_grad, b_grad


class Matmul(Function):
    '''t1 @ t2
    t1 @ t2 [2, 3] [3, 5] -> [2, 5]
    grad @ t2.T [2, 5] [5, 3] -> [2, 3]
    t1.T @ grad [3, 2] [2, 5] -> [3, 5]
    '''
    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        assert t1.ndim == t1.ndim, ''
        self.t1 = t1
        self.t2 = t2
        return t1 @ t2
    
    def backward(self, grad: Tensor) -> Tuple[Tensor]:
        return grad @ self.t2.T, self.t1.T @ grad


class GetItem(Function):
    '''getitem
    '''
    def __init__(self, index):
        self.index = index
        super().__init__()
    
    def forward(self, t: Tensor):
        self.t_shape = t.shape
        return t[self.index]
    
    def backward(self, grad):
        _grad = np.zeros(shape=self.t_shape)
        _grad[self.index] = grad
        return _grad




class Sum(Function):
    ''' sum 
    '''
    def __init__(self, axis, keepdims):
        if isinstance(axis, int):
            axis = (axis, )
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, t: Tensor):
        self.t_shape = t.shape
        return t.sum(self.axis, keepdims=self.keepdims)

    def backward(self, grad: Tensor):
        if self.axis is None:
            self.axis = tuple(range(len(self.t_shape)))

        if self.keepdims:
            shape = grad.shape
        else:
            shape = list(self.t_shape)
            for ax in self.axis:
                shape[ax] = 1
        
        return grad.reshape(shape) * np.ones(self.t_shape)


class Mean(Function):
    ''' mean
    '''
    def __init__(self, axis, keepdims):
        if isinstance(axis, int):
            axis = (axis, )
        self.axis = axis 
        self.keepdims = keepdims

    def forward(self, t: Tensor):
        self.t_shape = t.shape
        return t.mean(self.axis, keepdims=self.keepdims)
    
    def backward(self, grad: Tensor):
        if self.axis is None:
            self.axis = tuple(range(len(self.t_shape)))

        if self.keepdims:
            shape = grad.shape
        else:
            shape = list(self.t_shape)
            for ax in self.axis:
                shape[ax] = 1
        
        ks = [self.t_shape[i] for i in self.axis]
        return grad.reshape(shape) * np.ones(self.t_shape) / reduce(operator.mul, ks)


class Pow(Function):
    """pow 
    x^n -> n * (x ^ (n-1))
    n^x -> ln(y) = x*len(n) -> y' = y * ln(n)
    """
    def __init__(self, n):
        self.n = n 

    def forward(self, t):
        self.t = t
        return t ** self.n

    def backward(self, grad: Tensor):
        # grad * self.o * np.log(self.t + 1e-15)
        return grad * self.n * (self.t ** (self.n-1))


class Leaf(Function):
    '''leaf
    '''
    def __init__(self, variable, requires_grad):
        self.variable = variable
        self.output_ids = {id(variable): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad

    def _do_forward(self, *input):
        raise NotImplementedError

    def _do_backward(self, *grad_output):
        assert len(grad_output) == 1, ''
        if self.requires_grad:
            self.variable.grad += grad_output[0]
        return tuple()


