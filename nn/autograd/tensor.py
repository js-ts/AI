import numpy as np 
from functools import reduce
import operator
import copy

from dataclasses import dataclass
from typing import Union, Callable, List, NoReturn

# from .engine import ExecutionEngine


@dataclass
class Dependency:
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data.astype(np.float64)
    else:
        return np.array(data).astype(np.float64)

def to_tensor(data):
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data)


class Tensor:

    # _engine = ExecutionEngine()

    def __init__(self,
                data: Union[np.ndarray, list, float, int],
                requires_grad: bool=False,
                depends_on: List[Dependency] = None) -> None:
        
        self.data = to_numpy(data)        
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.grad: 'Tensor' = None
        self.shape = self.data.shape

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self, ):
        return f'Tensor({self.data}, requires_grad={self.requires_grad})'
        
    # # self.__class__
    def zero_grad(self, ) -> None:
        if self.grad == None:
            self.grad = Tensor(np.zeros(self.shape)) 
        else:
            self.grad.zero_()
        
    def backward(self, grad: 'Tensor'=None) -> 'Tensor':
        assert self.requires_grad, f'requires_grad={self.requires_grad}'

        # self._engine.run_backward(self, grad)     

        if grad is None:
            grad = Tensor(1.)

        self.grad.data += grad.data

        for depend in self.depends_on:
            _grad = depend.grad_fn(grad.data)
            depend.tensor.backward(Tensor(_grad))

    @property
    def is_leaf(self, ) -> bool:
        if self.depends_on:
            return False
        else:
            return True

    # magic method
    def __add__(self, other) -> 'Tensor':
        return op_add(self, to_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        return op_add(to_tensor(other), self)

    def __sub__(self, other) -> 'Tensor':
        return op_sub(self, to_tensor(other))
    
    def __rsub__(self, other) -> 'Tensor':
        return op_sub(to_tensor(other), self)

    def __neg__(self, ) -> 'Tensor':
        return op_neg(self)

    def __mul__(self, other) -> 'Tensor':
        return op_mul(self, to_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return op_mul(to_tensor(other), self)

    def __pow__(self, n: int) -> 'Tensor':
        return op_pow(self, n)

    def __matmul__(self, other) -> 'Tensor':
        return op_matmul(self, to_tensor(other))

    def __getitem__(self, idx) -> 'Tensor':
        return op_slice(self, idx)

    def __iadd__(self, other) -> None:
        raise NotImplementedError

    def __isub__(self, other) -> None:
        raise NotImplementedError

    def __abs__(self, ) -> None:
        raise NotImplementedError

    # method
    def sum(self, ) -> 'Tensor':
        return op_sum(self)    
    
    def mean(self, ) -> 'Tensor':
        return op_mean(self)

    # in-space
    def add_(self, other: 'Tensor') -> None:
        self.data += other.data

    def sub_(self, other: 'Tensor') -> None:
        self.data -= other.data

    def mul_(self, other: 'Tensor') -> None:
        self.data *= other.data

    def zero_(self, ) -> None:
        # self.data = np.zeros_like(self.data)
        self.data[...] = 0


def op_sum(t: Tensor) -> Tensor:
    '''
    '''
    data = t.data.sum()
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)
        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def op_mean(t: Tensor) -> Tensor:
    '''
    '''
    data = t.data.mean()
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data) / reduce(operator.mul, t.shape)
        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def op_add(t1: Tensor, t2: Tensor) -> Tensor:
    '''t1 + t2 = t
    '''
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            '''including broadcast'''
            _extend = grad.ndim - t1.data.ndim
            for _ in range(_extend):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            assert grad.shape == t1.shape, 'op_add'
            return grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            '''including broadcast'''
            _extend = grad.ndim - t2.data.ndim
            for _ in range(_extend):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            assert grad.shape == t2.shape, 'op_add'
            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def op_neg(t: Tensor) -> Tensor:
    '''
    '''
    data = -t.data
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return -grad
        depends_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)


def op_pow(t: Tensor, n: int) -> Tensor:
    '''x ^ n
    '''
    data = t.data ** n 
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (n * t.data ** (n-1))

        depends_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)


def op_sub(t1: Tensor, t2: Tensor) -> Tensor:
    '''
    '''
    return op_add(t1, op_neg(t2))


def op_mul(t1: Tensor, t2: Tensor) -> Tensor:
    '''t = t1 * t2
    '''
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            _grad = grad * t2.data
            _extend = _grad.ndim - t1.data.ndim
            for _ in range(_extend):
                _grad = _grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    _grad = _grad.sum(axis=i, keepdims=True)
            assert _grad.shape == t1.shape, 'op_mul'
            return _grad
        
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            _grad = grad * t1.data
            _extend = _grad.ndim - t2.data.ndim
            for _ in range(_extend):
                _grad = _grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    _grad = _grad.sum(axis=i, keepdims=True)
            assert _grad.shape == t2.shape, 'op_mul'
            return _grad
        depends_on.append(Dependency(t2, grad_fn2))
    
    return Tensor(data, requires_grad, depends_on)


def op_matmul(t1: Tensor, t2: Tensor) -> Tensor:
    '''t = t1 @ t2
    '''
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad @ t2.data.T
            return grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = t1.data.T @ grad
            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def op_slice(t: Tensor, idx):
    '''
    '''
    data = t.data[idx] # pointer # copy.deepcopy()
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            _grad = np.zeros_like(t.data)
            _grad[idx] = grad
            return _grad

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)


# functional

def tanh(t: Tensor) -> Tensor:
    data = np.tanh(t.data)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            '''tanh(x)
            '''
            return grad * (1 - data * data)
            
        depends_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)


def relu(t: Tensor) -> Tensor:
    mask = np.ones_like(t.data)
    mask[t.data < 0] = 0
    data = t.data * mask
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            '''relu(x)
            '''
            return grad * mask
        depends_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)


def sigmoid(t: Tensor) -> Tensor:

    data = 1. / (1. + np.exp(-t.data))
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return data * (1 - data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)




# TODO
# deepcopy
# func_params and return value
# reference

"""
def test_reference(x: np.ndarray) -> np.ndarray:
    _x = x
    return _x

x = np.random.rand(3, 3)
y = test_reference(x)
assert x is y, 'x is y'
"""