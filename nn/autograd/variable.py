import numpy as np
from numpy import ndarray as Tensor 

from typing import Any, Tuple, Optional
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
        # unpacked_input = []
        # for var in inputs:
        #     if isinstance(var, Variable):
        #         unpacked_input.append(var.data)
        #     else:
        #         unpacked_input.append(var)

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
            if _grad is not None:
                self._backward_fn(creator.previous_functions[_i][0], _grad)


# --- utils
def to_tensor(data):
    return np.array(data).astype(np.float64)

def to_variable(data):
    '''make sure data is variable'''
    if isinstance(data, (int, float)):
        data = to_tensor(data)
        return Variable(data)

    elif isinstance(data, (list, tuple)):
        data = to_tensor(data)
        return Variable(data)

    elif isinstance(data, Tensor):
        return Variable(data)
        
    elif isinstance(data, Variable):
        return data
# ---


class Variable(object):

    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        if creator is None:
            creator = Leaf(self, requires_grad)
        self.data = data
        self.creator = creator
        self.shape = self.data.shape
        self.requires_grad = self.creator.requires_grad
        
        self.grad = None
        if isinstance(creator, Leaf) and requires_grad:
            self.grad = np.zeros_like(data)

    def backward(self, grad=1.):
        self._engine._backward_fn(self.creator, grad)

    def register_hook(self, name, hook):
        self.creator.register_hook(name, hook)
    
    def remove_hook(self, name):
        self.creator.remove_hook(name)

    def reshape(self, *shape):
        return Reshape(*shape)(self)[0]


    def add(self, other):
        other = to_variable(other)
        return Add()(self, other)[0]
    
    def neg(self, ):
        return Neg()(self)[0]

    def sub(self, other):
        other = to_variable(other)
        # return self.add(other.neg())
        return Sub()(self, other)[0]

    def mul(self, other):
        other = to_variable(other)
        return Mul()(self, other)[0]

    def div(self, other):
        other = to_variable(other)
        return Div()(self, other)[0]

    def sum(self, ):
        return Sum()(self)[0]
    
    def mean(self, ):
        return Mean()(self)[0]

    def pow(self, n):
        n = to_variable(n)
        return Pow()(self, n)[0]

    def exp(self, ):
        return Exp()(self)[0]

    def sigmoid(self, ):
        return Sigmoid()(self)[0]

    def tanh(self, ):
        return Tanh()(self)[0]

    def matmul(self, other: 'Variable') -> 'Variable':
        return Matmul()(self, other)[0]
    
    # magic method
    def __add__(self, other):
        return self.add(other)

    def __neg__(self, ):
        return self.neg()

    def __sub__(self, other):
        return self.sub(other)

    def __pow__(self, n):
        return self.pow(n)
    
    def __div__(self, other):
        return self.div(other)
    __truediv__ = __div__

    def __mul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return self.matmul(other)
    
    # __radd__ = __add__
    def __radd__(self, other):
        other = to_variable(other)
        return other.add(self)
        
    def __rsub__(self, other):
        other = to_variable(other)
        return other.sub(self)
    
    def __rmul__(self, other):
        other = to_variable(other)
        return other.mul(self)

    def __rdiv__(self, other):
        other = to_variable(other)
        return other.div(self)
    __rtruediv__ = __rdiv__
    
    def __iadd__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __getitem__(self, idx):
        return Getitem(idx)(self)[0]


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
        # self.backward_hooks = OrderedDict()

    def _do_forward(self, *input):
        raise NotImplementedError

    def _do_backward(self, *grad_output):
        assert len(grad_output) == 1
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
    """add
    broadcast
    [1, 3] + [2, 4, 3] -> [2, 4, 3]
    """
    def forward(self, a, b):
        c = a + b
        self.a_shape = a.shape
        self.b_shape = b.shape
        self.c_shape = c.shape
        return c

    def backward(self, grad):
        assert self.c_shape == grad.shape, ''
        a_grad = broadcast_reverse(grad, self.a_shape)
        b_grad = broadcast_reverse(grad, self.b_shape)

        return a_grad, b_grad


class Neg(Function):
    """
    -t 
    """
    def forward(self, t: Tensor) -> Tensor:
        return -t 
    
    def backward(self, grad: Tensor) -> Tensor:
        return -grad


class Sub(Function):
    """a-b
    """
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a_shape = a.shape
        self.b_shape = b.shape
        return a - b

    def backward(self, grad):
        a_grad = broadcast_reverse( grad, self.a_shape)
        b_grad = broadcast_reverse(-grad, self.b_shape)
        return a_grad, b_grad 


class Mul(Function):
    """MUL"""
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        c = a * b
        self.a = a
        self.b = b
        self.c_shape = c.shape
        return c
    
    def backward(self, grad):
        assert self.c_shape == grad.shape
        a_grad = broadcast_reverse(grad * self.b, self.a.shape)
        b_grad = broadcast_reverse(grad * self.a, self.b.shape)

        return a_grad, b_grad


class Div(Function):
    """div
    """
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # np.testing.assert_almost_equal(b, 0)
        self.a = a
        self.b = b
        return a / (b + 1e-20)
    
    def backward(self, grad: Tensor):
        a_grad = grad / self.b
        b_grad = -grad * self.a / (self.b ** 2)
        return a_grad, b_grad


class Sum(Function):
    """ sum """
    def forward(self, t: Tensor):
        self.t = t
        return t.sum()

    def backward(self, grad: Tensor):
        print('mean')
        return grad * np.ones_like(self.t)


class Mean(Function):
    """ mean """
    def forward(self, t: Tensor):
        self.t = t
        return t.mean()
    
    def backward(self, grad: Tensor):
        return grad * np.ones_like(self.t) / reduce(operator.mul, self.t.shape)


class Pow(Function):
    """pow 
    x^n -> n * (x ^ (n-1))
    n^x -> ln(y) = x*len(n) -> y' = y * ln(n)
    """
    def forward(self, t, n):
        self.t = t
        self.n = n
        self.o = t ** n
        return self.o

    def backward(self, grad: Tensor):
        return grad * self.n * (self.t ** (self.n-1)), grad * self.o * np.log(self.t)


class Exp(Function):
    """exp """
    def forward(self, t: Tensor) -> Tensor:
        self.out = np.exp(t)
        return self.out
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.out


class Sigmoid(Function):
    """sigmoid """
    def forward(self, t: Tensor):
        self.out = 1. / (1. + np.exp(-t)) 
        return self.out
    
    def backward(self, grad):
        return grad * self.out / (1. - self.out + 1e-10)


class ReLU(Function):
    """relu """ 
    def forward(self, t: Tensor) -> Tensor:
        self.mask = t > 0
        return t * self.mask
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.mask


class Tanh(Function):
    """
    formul: (exp(x) + exp(-x)) / (exp(x) - exp(-x))
    derive : 1 - tanh(x) ** 2
    """
    def forward(self, t: Tensor) -> Tensor:
        self.out = np.tanh(t)
        return self.out
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad  * (1 - self.out ** 2)


class Matmul(Function):
    """
    t1 @ t2 [2, 3] [3, 5] -> [2, 5]
    grad @ t2.T [2, 5] [5, 3] -> [2, 3]
    t1.T @ grad [3, 2] [2, 5] -> [3, 5]
    """
    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        self.t1 = t1
        self.t2 = t2
        return t1 @ t2
    
    def backward(self, grad: Tensor) -> Tuple[Tensor]:
        return grad @ self.t2.T, self.t1.T @ grad
    

class Getitem(Function):
    """getitem"""
    def __init__(self, index):
        self.index = index
        super().__init__()
    
    def forward(self, t: Tensor):
        self.t_shape = t.shape
        return t[self.index]
    
    def backward(self, grad):
        _grad = np.zeros(shape=self.t_shape)
        _grad[self.index] = grad
        return _grad


class Reshape(Function):
    def __init__(self, *shape):
        self.shape = shape
        super().__init__()
    
    def forward(self, t: Tensor) -> Tensor:
        self.t_shape = t.shape
        return t.reshape(*self.shape)
    
    def backward(self, grad: Tensor) -> Tensor:
        grad = grad[...]
        return grad.reshape(*self.t_shape)