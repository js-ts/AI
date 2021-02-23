from typing import Tuple, Iterable, Iterator, Union

from functools import reduce as REDUCE
from operator import mul as MUL

from pdll.backend import np
from pdll.backend import Tensor

from .function import Function
from .variable import Variable
from .backpropag import Leaf

from .utils import broadcast_reverse
from .utils import register
from .utils import to_variable


__all__ = [
    'add', 'sub',
]

class _Add(Function):
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


class _Sub(Function):
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


class _Neg(Function):
    '''-t 
    '''
    def forward(self, t: Tensor) -> Tensor:
        return -t 
    
    def backward(self, grad: Tensor) -> Tensor:
        return -grad


class _Mul(Function):
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


class _Div(Function):
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


class _Matmul(Function):
    '''t1 @ t2
    t1 @ t2 [2, 3] [3, 5] -> [2, 5]
    grad @ t2.T [2, 5] [5, 3] -> [2, 3]
    t1.T @ grad [3, 2] [2, 5] -> [3, 5]
    '''
    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        # assert t1.ndim == t1.ndim, ''
        out = t1 @ t2
        self.t1 = t1
        self.t2 = t2
        self.out_shape = out.shape
        return out
    
    def backward(self, grad: Tensor) -> Tuple[Tensor]:
        assert grad.shape == self.out_shape, ''

        t1_shape = self.t1.shape
        t2_shape = self.t2.shape
        t1_shape_t_idx = list(range(len(t1_shape)-2)) + list(range(len(t1_shape)-2, len(t1_shape)))[::-1]
        t2_shape_t_idx = list(range(len(t2_shape)-2)) + list(range(len(t2_shape)-2, len(t2_shape)))[::-1]
        
        grad_t1 = grad @ self.t2.transpose(t2_shape_t_idx)
        grad_t2 = self.t1.transpose(t1_shape_t_idx) @ grad

        grad_t1 = broadcast_reverse(grad_t1, self.t1.shape)
        grad_t2 = broadcast_reverse(grad_t2, self.t2.shape)

        return grad_t1, grad_t2


class _GetItem(Function):
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



class _Sum(Function):
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


class _Mean(Function):
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
        return grad.reshape(shape) * np.ones(self.t_shape) / REDUCE(MUL, ks)


class _Pow(Function):
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


class _Reshape(Function):
    def __init__(self, *shape):
        self.shape = shape
        super().__init__()
    
    def forward(self, t: Tensor) -> Tensor:
        self.t_shape = t.shape
        return t.reshape(*self.shape)
    
    def backward(self, grad: Tensor) -> Tensor:
        grad = grad[...]
        return grad.reshape(*self.t_shape)


class _Transpose(Function):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, t: Tensor):
        assert len(self.dims) == len(t.shape)
        return t.transpose(*self.dims)
    
    def backward(self, grad: Tensor):
        idx_reverse = np.argsort(self.dims)
        return grad.transpose(*idx_reverse)


class _Exp(Function):
    """exp 
    """
    def forward(self, t: Tensor) -> Tensor:
        self.out = np.exp(t)
        return self.out
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.out


class _RPow(Function):
    '''a ** x
    (a ** x) * log(a)
    '''
    def __init__(self, a):
        self.a = a

    def forward(self, t: Tensor):
        self.t = t
        self.out = self.a ** t 
        return self.out
    
    def backward(self, grad: Tensor):
        return grad * self.out * np.log(self.a + 1e-10)


# ------ register method

@register(Variable)
def add(self, other):
    other = to_variable(other)
    return _Add()(self, other)[0]

@register(Variable)
def sub(self, other):
    other = to_variable(other)
    return _Sub()(self, other)[0]

@register(Variable)
def neg(self, ):
    return _Neg()(self)[0]

@register(Variable)
def mul(self, other):
    other = to_variable(other)
    return _Mul()(self, other)[0]

@register(Variable)
def div(self, other):
    other = to_variable(other)
    return _Div()(self, other)[0]

@register(Variable)
def matmul(self, other):
    # other = to_variable(other)
    return _Matmul()(self, other)[0]

@register(Variable)
def pow(self, n):
    return _Pow(n)(self)[0]

@register(Variable)
def sqrt(self, ):
    return _Pow(1/2.)(self)[0]

@register(Variable)    
def exp(self, ):
    return _Exp()(self)[0]

@register(Variable)
def rpow(self, a):
    return _RPow(a)(self)[0]

@register(Variable)
def sum(self, axis=None, keepdims=False):
    return _Sum(axis, keepdims)(self)[0]

@register(Variable)
def mean(self, axis=None, keepdims=False):
    return _Mean(axis, keepdims)(self)[0]

# TODO
@register(Variable)
def var(self, axis=None, keepdims=False):
    return ((self - self.mean(axis, True)) ** 2).mean(axis, keepdims)

@register(Variable)
def reshape(self, *shape):
    return _Reshape(*shape)(self)[0]

@register(Variable)
def transpose(self, *dims):
    return _Transpose(*dims)(self)[0]

# magic-method
@register(Variable)
def __add__(self, other):
    '''self + other
    '''
    return self.add(other)

@register(Variable)
def __radd__(self, other):
    '''other + self
    '''
    other = to_variable(other)
    return other.add(self)

@register(Variable)
def __sub__(self, other):
    return self.sub(other)

@register(Variable)
def __rsub__(self, other):
    other = to_variable(other)
    return other.sub(self)

@register(Variable)
def __neg__(self, ):
    return self.neg()

@register(Variable)
def __mul__(self, other):
    return self.mul(other)

@register(Variable)
def __rmul__(self, other):
    other = to_variable(other)
    return other.mul(self)

@register(Variable)
def __div__(self, other):
    return self.div(other)

@register(Variable)
def __truediv__(self, other):
    return self.div(other)

@register(Variable)
def __rdiv__(self, other):
    other = to_variable(other)
    return other.div(self)

@register(Variable)
def __rtruediv__(self, other):
    other = to_variable(other)
    return other.div(self)

@register(Variable)
def __matmul__(self, other):
    return self.matmul(other)

@register(Variable)
def __pow__(self, n):
    return self.pow(n)

@register(Variable)
def __rpow__(self, a):
    return self.rpow(a)

@register(Variable)
def __getitem__(self, idx):
    return _GetItem(idx)(self)[0]



# ---- inspace op

def gaurantee_inspace(var: Variable):
    '''
    '''
    if isinstance(var.creator, Leaf) and var.requires_grad is True:
        raise RuntimeError('')

@register(Variable)
def zeros_(self, ):
    gaurantee_inspace(self)
    self.tensor[...] = 0

@register(Variable)
def add_(self, other: Union['Variable', Tensor]) -> None:
    gaurantee_inspace(self)
    if isinstance(other, Variable):
        self.tensor[...] += other.tensor
    elif isinstance(other, Tensor):
        self.tensor[...] += other

@register(Variable)
def sub_(self, other: Union['Variable', Tensor]) -> None:
    gaurantee_inspace(self)
    if isinstance(other, Variable):
        self.tensor[...] -= other.tensor
    elif isinstance(other, Tensor):
        self.tensor[...] -= other

@register(Variable)
def mul_(self, other: Union['Variable', Tensor]) -> None:
    gaurantee_inspace(self)
    if isinstance(other, Variable):
        self.tensor[...] *= other.tensor
    elif isinstance(other, Tensor):
        self.tensor[...] *= other