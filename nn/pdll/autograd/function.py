from collections import OrderedDict

from .variable import Variable

class Function(object):

    def __init__(self, ):
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None

    def _do_forward(self, *inputs):
        '''
        '''
        # engine._backward_var
        # self.inputs = inputs 

        unpacked_input = []
        needs_input_grad = []
        for var in inputs:
            if isinstance(var, Variable):
                unpacked_input.append(var.data)
                needs_input_grad.append(var.creator.requires_grad)
            else:
                unpacked_input.append(var)
                needs_input_grad.append(False)

        raw_output = self.forward(*unpacked_input)

        if not isinstance(raw_output, tuple):
            raw_output = (raw_output, )
        
        self.needs_input_grad = needs_input_grad
        self.requires_grad = any(self.needs_input_grad)
        self.previous_functions = [(arg.creator, id(arg)) for arg in inputs]

        output = tuple(Variable(data, self) for data in raw_output)

        self.output_ids = {id(var): i for i, var in enumerate(output)}

        return output

    __call__ = _do_forward

    def _do_backward(self, grad_output):
        '''
        '''
        grad_input = self.backward(grad_output) 
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input, )
        
        assert len(grad_input) == len(self.previous_functions), f'{self.__class__.__name__}'

        return grad_input

    
    def forward(self, *inputs):
        '''tensor -> tensor
        '''
        raise NotImplementedError

    def backward(self, *grad_output):
        '''tensor -> tensor
        '''
        raise NotImplementedError

