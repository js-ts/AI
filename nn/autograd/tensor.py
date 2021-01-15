import numpy as np 
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
        self.grad = self.__class__(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor'=None) -> 'Tensor':
        assert self.requires_grad, f'requires_grad={self.requires_grad}'
        
        if grad is None:
            grad = self.__class__(1.)

        self.grad.data += grad.data

        for depend in self.depends_on:
            _grad = depend.grad_fn(grad.data)
            depend.tensor.backward(Tensor(_grad))

    
    def sum(self, ) -> 'Tensor':
        return op_sum(self)    
    


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
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad
        depends_on.append(Dependency(t1, grad_fn))

    if t2.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad
        depends_on.append(Dependency(t2, grad_fn))

    return Tensor(data, requires_grad, depends_on)