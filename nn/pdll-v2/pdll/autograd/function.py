from collections import OrderedDict
from typing import Union, Tuple, List, Any, Iterable

from pdll.backend import Tensor
from .variable import Variable


class Function(object):
    '''function
    '''
    def __init__(self, ):
        self.previous_functions = None
        self.output_ids = None
        self.input_ids = None
        self.needs_input_grad = None

    def _do_forward(self, *inputs: Any) -> Iterable[Variable]:
        '''Tuple[Varible, ...]
        '''
        unpacked_input = []
        needs_input_grad = []
        previous_functions = []
        for var in inputs:
            if isinstance(var, Variable):
                unpacked_input.append(var.data)
                needs_input_grad.append(var.creator.requires_grad)
                previous_functions.append((var.creator, id(var)))
            else:
                # print('-----_do_forward-----', self.__class__.__name__)
                unpacked_input.append(var)
                needs_input_grad.append(False)
                previous_functions.append((None, -1))
        
        self.needs_input_grad = needs_input_grad
        self.requires_grad = any(needs_input_grad)
        self.previous_functions = previous_functions

        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output, )

        output = tuple(Variable(data, creator=self) for data in raw_output)

        self.input_ids = {id(var): i for i, var in enumerate(inputs)}
        self.output_ids = {id(var): i for i, var in enumerate(output)}

        return output

    __call__ = _do_forward

    def _do_backward(self, output_grad: Tensor) -> Tuple[Tensor, ...]:
        '''
        '''
        grad_inputs = self.backward(output_grad) 
        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs, )
        
        assert len(grad_inputs) == len(self.previous_functions), f'{self.__class__.__name__} _do_backward'

        return grad_inputs


    def forward(self, *inputs):
        '''tensor -> tensor
        '''
        raise NotImplementedError


    def backward(self, grad_output):
        '''tensor -> tensor
        '''
        raise NotImplementedError



