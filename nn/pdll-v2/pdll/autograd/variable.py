from typing import Union
from ..backend import Tensor
from ..backend import np


class ExecuteEngine(object):
    
    def __init__(self, debug=False):
        self.debug = debug

    def build_graph(self, ):
        '''
        '''
        raise NotImplementedError

    def backward_var(self, var, grad) -> None:
        ''' '''
        var.grad += grad
        grads_input = var.creator._do_backward(grad)
        for _i, _grad in enumerate(grads_input):
            if _grad is not None:
                self.backward_var(var.creator.inputs[_i], _grad)

    def backward_fn(self, creator, grad) -> None:
        ''' '''
        grads_input = creator._do_backward(grad)
        for _i, _grad in enumerate(grads_input):
            if _grad is not None:
                self.backward_fn(creator.previous_functions[_i][0], _grad)


class Leaf(object):
    '''leaf
    '''
    def __init__(self, variable, requires_grad):
        self.variable = variable
        self.output_ids = {id(variable): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad

    def _do_backward(self, *grad_output):
        assert len(grad_output) == 1, ''
        if self.requires_grad:
            if self.variable.grad is None:
                self.variable.grad = np.zeros(self.variable.shape)
            self.variable.grad[...] += grad_output[0]
        return tuple()


class Variable(object):
    '''
    '''
    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        assert isinstance(data, (Tensor, np.float32, np.float64)), f'{data} {type(data)}'
        assert isinstance(requires_grad, bool), ''

        if creator is None:
            creator = Leaf(self, requires_grad)
        self.creator = creator

        self.data = data
        self.grad = None
        self.shape = self.data.shape
        self.requires_grad = self.creator.requires_grad

    def backward(self, grad=None):
        '''backward
        '''
        if grad is None:
            grad = np.ones_like(self.data)
        elif isinstance(grad, (int, float)):
            grad = np.array([grad])
        else:
            assert isinstance(grad, Tensor) and grad.shape == self.shape

        self._engine.backward_fn(self.creator, grad)

    def zero_grad(self, ):
        # assert self.grad is not None, ''
        if self.grad is not None:
            self.grad[...] = 0

    def register_hook(self, name, hook):
        raise NotImplementedError

    def remove_hook(self, name):
        raise NotImplementedError 

    def __repr__(self, ):
        return f'Variable(data={self.data}, requires_grad={self.requires_grad})'

    # def __setattr__(self, name, value):
    #     if all([name == 'requires_grad', value, isinstance(self.creator, Leaf)]):
    #         self.grad = np.zeros(self.shape)
    #     object.__setattr__(self, name, value)

    def zeros_(self, ):
        self.data[...] = 0
    
    def add_(self, other: Union['Variable', Tensor]) -> None:
        if isinstance(other, Variable):
            self.data[...] += other.data
        elif isinstance(other, Tensor):
            self.data[...] += other

    def sub_(self, other: Union['Variable', Tensor]) -> None:
        if isinstance(other, Variable):
            self.data[...] -= other.data
        elif isinstance(other, Tensor):
            self.data[...] -= other

    def mul_(self, other: Union['Variable', Tensor]) -> None:
        if isinstance(other, Variable):
            self.data[...] *= other.data
        elif isinstance(other, Tensor):
            self.data[...] *= other