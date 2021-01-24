import numpy as np
import numpy.ndarray as Tensor 

# from .tensor import Tensor
# from .tensor import Dependency

from typing import Any
from collections import OrderedDict


class _ContextMixin(object):

    def save_for_backward(self, *tensors):
        self.to_save = tensors
    

class _BaseFunction(object):
    pass



class _Function(_BaseFunction, _ContextMixin):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError



class Variable(object):
    def __init__(self, data, creator=None, requires_grad=True):
        if creator:
            self.creator = Leaf(data, requires_grad)
        self.data = data
        self.creator = creator
        self._grad = None    



class Function(object):

    def __init__(self, ):
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None
        self.backward_hooks = OrderedDict()
    
    __call__ = _do_forward

    def _do_forward(self, *inputs):
        '''
        '''
        unpacked_input = tuple(arg.data for arg in inputs)
        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output, )
        
        self.needs_input_grad = tuple(arg.requires_grad for arg in inputs)
        self.requires_grad = any(self.needs_input_grad)

        output = tuple(Variable(data, self) for data in raw_output)

        self.previous_functions = [(arg.creator, id(arg)) for arg in inputs]
        self.output_ids = {id(var): i for i, var in enumerate(output)}

        return output


    def _do_backward(self, grad_output):
        '''
        '''
        grad_input = self.backward(grad_output) 
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input, )
        
        assert len(grad_input) == len(self.previous_functions), f'{self.__class__.__name__}'

        # for hook, idx in self.backward_hooks.values():
        return grad_input


    def register_hook(self, name, hook, tensor):
        assert name not in self.backward_hooks, ''
        ids = self.output_ids[id(tensor)] if tensor else None
        self.backward_hooks[name] = (hook, ids)

    def remove_hook(self, name):
        assert name in self.backward_hooks, ''
        del self.backward_hooks[name]
    

    def forward(self, *inputs):
        '''ndarray -> ndarray
        '''
        raise NotImplementedError

    def backward(self, *grad_output):
        '''ndarray -> ndarray
        '''
        raise NotImplementedError



class Leaf(Function):

    def __init__(self, variable, requires_grad):
        self.variable = variable
        self.output_ids = {id(variable): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad
        self.backward_hooks = OrderedDict()

    def _do_forward(self, *input):
        raise NotImplementedError

    def _do_backward(self, *grad_output):
        assert len(grad_output) == 1
        for hook in self.backward_hooks.values():
            hook(grad_output, grad_output)
        self.variable.grad.add_(grad_output[0])
        return tuple()

