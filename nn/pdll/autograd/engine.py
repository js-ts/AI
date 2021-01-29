from typing import Callable
from .variable import Variable

class ExecuteEngine(object):
    
    def __init__(self, ):
        pass

    def _backward_var(self, var, grad):
        ''' '''
        var.grad += grad
        grads_input = var.creator._do_backward(grad)
        for _i, _grad in enumerate(grads_input):
            if _grad is not None:
                self._backward_var(var.creator.inputs[_i], _grad)

    def _backward_fn(self, creator, grad):
        ''' '''
        grads_input = creator._do_backward(grad.data)
        for _i, _grad in enumerate(grads_input):
            if _grad is not None:
                self._backward_fn(creator.previous_functions[_i][0], _grad)

