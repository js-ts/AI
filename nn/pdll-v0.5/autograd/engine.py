# from .variable import Variable
# from .function import Function
# from .tensor import Tensor

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

