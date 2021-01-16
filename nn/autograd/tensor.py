import numpy as np 
import copy

from dataclasses import dataclass

from typing import Union, Callable, List


@dataclass
class Dependency:
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    else:
        return np.array(data).astype(np.float32)

def to_tensor(data):
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data)


class Tensor:
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
        
    def zero_grad(self, ) -> None:
        self.grad = Tensor(np.zeros(self.shape)) # self.__class__

    def backward(self, grad: 'Tensor'=None) -> 'Tensor':
        assert self.requires_grad, f'requires_grad={self.requires_grad}'
        
        if grad is None:
            grad = Tensor(1.)

        self.grad.data += grad.data

        for depend in self.depends_on:
            _grad = depend.grad_fn(grad.data)
            depend.tensor.backward(Tensor(_grad))
    
    def sum(self, ) -> 'Tensor':
        return op_sum(self)    

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

    def __matmul__(self, other) -> 'Tensor':
        return op_matmul(self, to_tensor(other))

    def __getitem__(self, idx) -> 'Tensor':
        return op_slice(self, idx)

    def __iadd__(self, other) -> None:
        pass
    def __isub__(self, other) -> None:
        pass

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
            return _grad
        depends_on.append(Dependency(t2, grad_fn2))
    
    return Tensor(data, requires_grad, depends_on)


def op_matmul(t1: Tensor, t2: Tensor) -> Tensor:
    '''t = t1 @ t2
    '''
    print(t1.shape, t2.shape)
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            _grad = grad @ t2.data.T
            return _grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            _grad = t1.data.T @ grad
            return _grad
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