
from pdll.backend import np 

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
                self.variable.grad = type(self.variable)(np.zeros(self.variable.shape))
            self.variable.grad.tensor += grad_output[0].tensor
        return tuple()
