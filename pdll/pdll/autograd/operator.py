from typing import Tuple, Iterable, Iterator, Union

from functools import reduce as REDUCE
from operator import mul as MUL

from pdll.backend import np
from pdll.backend import support_types

from .function import Function
from .tensor import Tensor
from .backpropag import Leaf

from .utils import broadcast_reverse
from .utils import register
from .utils import to_tensor


__all__ = [
    'add', 'sub', 'mul', 'neg', 'div', 'matmul', 'pow', 'exp',
    'sum', 'mean',
]

class _Add(Function):
    '''a + b
    add broadcast
    [1, 3] + [2, 4, 3] -> [2, 4, 3]
    '''
    def forward(self, a: Union[support_types], b: Union[support_types]) -> Union[support_types]:
        c = a + b
        self.a_shape = a.shape
        self.b_shape = b.shape
        self.c_shape = c.shape
        return c

    def backward(self, grad: Union[support_types]) -> Tuple[Union[support_types], Union[support_types]]:
        assert self.c_shape == grad.shape, 'add' 
        a_grad = broadcast_reverse(grad, self.a_shape)
        b_grad = broadcast_reverse(grad, self.b_shape)
        return a_grad, b_grad


class _Sub(Function):
    '''a - b
    '''
    def forward(self, a: Union[support_types], b: Union[support_types]) -> Union[support_types]:
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
    def forward(self, t: Union[support_types]) -> Union[support_types]:
        return -t 
    
    def backward(self, grad: Union[support_types]) -> Union[support_types]:
        return -grad


class _Mul(Function):
    '''a * b
    '''
    def forward(self, a: Union[support_types], b: Union[support_types]) -> Union[support_types]:
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
    def forward(self, a: Union[support_types], b: Union[support_types], eps: float=1e-10) -> Union[support_types]:
        # np.testing.assert_almost_equal(b, 0)
        c = a / b
        self.a = a
        self.b = b
        self.c_shape = c.shape
        self.eps = eps
        return c
    
    def backward(self, grad: Union[support_types]):
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
    def forward(self, t1: Union[support_types], t2: Union[support_types]) -> Union[support_types]:
        # assert t1.ndim == t1.ndim, ''
        out = t1 @ t2
        self.t1 = t1
        self.t2 = t2
        self.out_shape = out.shape
        return out
    
    def backward(self, grad: Union[support_types]) -> Tuple[Union[support_types]]:
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
    
    def forward(self, t: Union[support_types]):
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

    def forward(self, t: Union[support_types]):
        self.t_shape = t.shape
        return t.sum(self.axis, keepdims=self.keepdims)

    def backward(self, grad: Union[support_types]):
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

    def forward(self, t: Union[support_types]):
        self.t_shape = t.shape
        return t.mean(self.axis, keepdims=self.keepdims)
    
    def backward(self, grad: Union[support_types]):
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

    def backward(self, grad: Union[support_types]):
        # grad * self.o * np.log(self.t + 1e-15)
        return grad * self.n * (self.t ** (self.n-1))


class _Reshape(Function):
    def __init__(self, *shape):
        self.shape = shape
        super().__init__()
    
    def forward(self, t: Union[support_types]) -> Union[support_types]:
        self.t_shape = t.shape
        return t.reshape(*self.shape)
    
    def backward(self, grad: Union[support_types]) -> Union[support_types]:
        grad = grad[...]
        return grad.reshape(*self.t_shape)


class _Transpose(Function):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, t: Union[support_types]):
        assert len(self.dims) == len(t.shape)
        return t.transpose(*self.dims)
    
    def backward(self, grad: Union[support_types]):
        idx_reverse = np.argsort(self.dims)
        return grad.transpose(*idx_reverse)


class _Exp(Function):
    """exp 
    """
    def forward(self, t: Union[support_types]) -> Union[support_types]:
        self.out = np.exp(t)
        return self.out
    
    def backward(self, grad: Union[support_types]) -> Union[support_types]:
        return grad * self.out


class _RPow(Function):
    '''a ** x
    (a ** x) * log(a)
    '''
    def __init__(self, a):
        self.a = a

    def forward(self, t: Union[support_types]):
        self.t = t
        self.out = self.a ** t 
        return self.out
    
    def backward(self, grad: Union[support_types]):
        return grad * self.out * np.log(self.a + 1e-10)


# ------ register method

@register()
def add(self, other):
    other = to_tensor(other)
    return _Add()(self, other)[0]

@register()
def sub(self, other):
    other = to_tensor(other)
    return _Sub()(self, other)[0]

@register()
def neg(self, ):
    return _Neg()(self)[0]

@register()
def mul(self, other):
    other = to_tensor(other)
    return _Mul()(self, other)[0]

@register()
def div(self, other):
    other = to_tensor(other)
    return _Div()(self, other)[0]

@register()
def matmul(self, other):
    # other = to_tensor(other)
    return _Matmul()(self, other)[0]

@register()
def pow(self, n):
    return _Pow(n)(self)[0]

@register()
def sqrt(self, ):
    return _Pow(1/2.)(self)[0]

@register()    
def exp(self, ):
    return _Exp()(self)[0]

@register()
def rpow(self, a):
    return _RPow(a)(self)[0]

@register()
def sum(self, axis=None, keepdims=False):
    return _Sum(axis, keepdims)(self)[0]

@register()
def mean(self, axis=None, keepdims=False):
    return _Mean(axis, keepdims)(self)[0]

# TODO
@register()
def var(self, axis=None, keepdims=False):
    return ((self - self.mean(axis, True)) ** 2).mean(axis, keepdims)

@register()
def reshape(self, *shape):
    return _Reshape(*shape)(self)[0]

@register()
def transpose(self, *dims):
    return _Transpose(*dims)(self)[0]

# magic-method
@register()
def __add__(self, other):
    '''self + other
    '''
    return self.add(other)

@register()
def __radd__(self, other):
    '''other + self
    '''
    other = to_tensor(other)
    return other.add(self)

@register()
def __sub__(self, other):
    return self.sub(other)

@register()
def __rsub__(self, other):
    other = to_tensor(other)
    return other.sub(self)

@register()
def __neg__(self, ):
    return self.neg()

@register()
def __mul__(self, other):
    return self.mul(other)

@register()
def __rmul__(self, other):
    other = to_tensor(other)
    return other.mul(self)

@register()
def __div__(self, other):
    return self.div(other)

@register()
def __truediv__(self, other):
    return self.div(other)

@register()
def __rdiv__(self, other):
    other = to_tensor(other)
    return other.div(self)

@register()
def __rtruediv__(self, other):
    other = to_tensor(other)
    return other.div(self)

@register()
def __matmul__(self, other):
    return self.matmul(other)

@register()
def __pow__(self, n):
    return self.pow(n)

@register()
def __rpow__(self, a):
    return self.rpow(a)

@register()
def __getitem__(self, idx):
    return _GetItem(idx)(self)[0]



# ---- inspace-op

def gaurantee_inspace(var: Tensor):
    '''
    '''
    if isinstance(var.creator, Leaf) and var.requires_grad is True:
        raise RuntimeError('')

@register()
def zeros_(self, ):
    gaurantee_inspace(self)
    self.storage[...] = 0

@register()
def add_(self, other) -> None:
    gaurantee_inspace(self)
    if isinstance(other, ):
        self.storage[...] += other.storage
    elif isinstance(other, Union[support_types]):
        self.storage[...] += other

@register()
def sub_(self, other) -> None:
    gaurantee_inspace(self)
    if isinstance(other, ):
        self.storage[...] -= other.storage
    elif isinstance(other, Union[support_types]):
        self.storage[...] -= other

@register()
def mul_(self, other) -> None:
    gaurantee_inspace(self)
    if isinstance(other, ):
        self.storage[...] *= other.storage
    elif isinstance(other, Union[support_types]):
        self.storage[...] *= other