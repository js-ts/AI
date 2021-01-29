from typing import Tuple

from .tensor import Tensor
from .function import Function


class Leaf(Function):

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



def broadcast_reverse(grad: Tensor, shape: Tuple[int]) -> Tensor: 
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
    '''add broadcast
    [1, 3] + [2, 4, 3] -> [2, 4, 3]
    '''
    def forward(self, a, b):
        c = a + b
        self.a_shape = a.shape
        self.b_shape = b.shape
        self.c_shape = c.shape
        return c

    def backward(self, grad):
        assert self.c_shape == grad.shape, 'add' 
        a_grad = broadcast_reverse(grad, self.a_shape)
        b_grad = broadcast_reverse(grad, self.b_shape)
        return a_grad, b_grad