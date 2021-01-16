import numpy as np 
from typing import Dict, Callable

from .tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, data: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    '''data @ w + b'''
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.params['w'] = np.random.rand(input_dim, output_dim)
        self.params['b'] = np.random.rand(output_dim)

    def forward(self, data: Tensor) -> Tensor:
        self.data = data
        return data @ self.params['w'] + self.params['b']

    def backward(self, grad: Tensor) -> Tensor:
        '''
        y = f(x) and x = a @ w + b
        dy/da = f'(x) @ w.T
        dy/dw = a.T @ f'(x)
        dy/db = f'(x)
        '''
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.data.T @ grad

        return grad @ self.params['w'].T 
    

F = Callable[[Tensor], Tensor]

class Activition(Layer):
    '''
    '''
    def __init__(self, f: F, f_prime: F):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, data):
        self.data = data
        return self.f(data)
    
    def backward(self, grad):
        return grad * self.f_prime(self.data)



def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activition):
    def __init__(self, ):
        super().__init__(tanh, tanh_prime)
    