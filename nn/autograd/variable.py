import numpy as np
from numpy import ndarray as Tensor 

from typing import Any
from collections import OrderedDict, Counter

from functools import reduce
import operator


class Function(object):

    def __init__(self, ):
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None
        self.backward_hooks = OrderedDict()

    def _do_forward(self, *inputs):
        '''
        '''
        self.inputs = inputs # for backword

        unpacked_input = tuple(arg.data for arg in inputs)
        raw_output = self.forward(*unpacked_input)

        if not isinstance(raw_output, tuple):
            raw_output = (raw_output, )
        
        self.needs_input_grad = tuple(arg.creator.requires_grad for arg in inputs)
        self.requires_grad = any(self.needs_input_grad)

        output = tuple(Variable(data, self) for data in raw_output)

        self.previous_functions = [(arg.creator, id(arg)) for arg in inputs]
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


    def register_hook(self, name, hook, tensor):
        assert name not in self.backward_hooks, ''
        ids = self.output_ids[id(tensor)] if tensor else None
        self.backward_hooks[name] = (hook, ids)

    def remove_hook(self, name):
        assert name in self.backward_hooks, ''
        del self.backward_hooks[name]
    
    def forward(self, *inputs):
        '''tensor -> tensor
        '''
        raise NotImplementedError

    def backward(self, *grad_output):
        '''tensor -> tensor
        '''
        raise NotImplementedError



class ExecuteEngine(object):
    
    def __init__(self, ):
        pass

    def _compute_depends(self, function):
        depends = {}
        seen = {function}
        queue = [function]

        while len(queue) > 0:

            fn = queue.pop()
            for prev_fn, arg_id in fn.previous_functions:
                if prev_fn not in depends:
                    depends[prev_fn] = [Counter() for _ in prev_fn.output_ids]
                output_idx = prev_fn.output_ids[arg_id]
                depends[prev_fn][output_idx][fn] += 1
                if prev_fn not in seen:
                    queue.append(prev_fn)
                    seen.add(prev_fn)

        return depends


    def _free_backward_depend(self, depends, prev_fn, fn, arg_id):
        deps = depends[prev_fn]
        output_idx = prev_fn.output_ids[arg_id]
        output_deps = deps[output_idx]
        output_deps[fn] -= 1
        if output_deps[fn] == 0:
            del output_deps[fn]
        return output_idx


    def _is_ready_for_backward(self, depends, functiton):
        for deps in depends[functiton]:
            if len(deps) > 0:
                return False
        return True


    def run_backward(self, variable, grad):
        
        ready = [(variable.creator, (grad, ))]
        not_ready = []

        depends = self._compute_depends(variable.creator)
        
        while len(ready) > 0:
            
            fn, grad = ready.pop()
            grad_input = fn._do_backward(*grad)

            for (prev_fn, arg_id), d_prev_fn in zip(fn.previous_functions, grad_input):
                if not prev_fn.requires_grad:
                    assert d_prev_fn is None
                    continue

            output_nr = self._free_backward_depend(depends, prev_fn, fn, arg_id)
            is_ready = self._is_ready_for_backward(depends, prev_fn)
                
        raise NotImplementedError


    def _backward_var(self, var, grad):
        ''' '''
        var.grad += grad
        grads_input = var.creator._do_backward(grad)
        for _i, _grad in enumerate(grads_input):
            # var.creator.previous_functions[_i][0]._do_backward(_grad)
            self._backward_var(var.creator.inputs[_i], _grad)

    def _backward_fn(self, creator, grad):
        grads_input = creator._do_backward(grad)
        for _i, _grad in enumerate(grads_input):
            self._backward_fn(creator.previous_functions[_i][0], _grad)


class Variable(object):

    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        if creator is None:
            creator = Leaf(self, requires_grad)
        self.data = data
        self.creator = creator
        self.grad = None

        if isinstance(creator, Leaf) and requires_grad:
            self.grad = np.zeros_like(data)

    def backward(self, grad):
        if grad is None:
            grad = 1.
        self._engine._backward_fn(self.creator, grad)

    def add(self, other):
        return Add()(self, other)[0]

    def sum(self, ):
        return Sum()(self)[0]
    
    def mean(self, ):
        return Mean()(self)[0]

    def __add__(self, other):
        return self.add(other)

# ====
class Parameter(Variable):

    def __init__(self, *shape):
        data = np.random.rand(shape) * 2 - 1
        super().__init__(data, requires_grad=True)


# ====
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
        if self.requires_grad:
            self.variable.grad += grad_output[0]
        return tuple()


class Add(Function):

    def forward(self, a, b):
        return a + b

    def backward(self, grad):
        return grad, grad


class Sum(Function):

    def forward(self, t: Tensor):
        self.t = t
        return t.sum()

    def backward(self, grad: Tensor):
        print('mean')
        return grad * np.ones_like(self.t)


class Mean(Function):

    def forward(self, t: Tensor):
        self.t = t
        return t.mean()
    
    def backward(self, grad: Tensor):
        return grad * np.ones_like(self.t) / reduce(operator.mul, self.t.shape)

