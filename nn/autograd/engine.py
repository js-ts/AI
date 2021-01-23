from .tensor import Tensor

from dataclasses import dataclass
from typing import List, Tuple, Callable

class ExecutionEngine:

    def run_backward(self, t: Tensor, grad: Tensor):
        raise NotImplementedError



@dataclass
class Denpend_on:
    inputs: List[Tensor]
    grad_fn: Callable[[Tensor], Tuple[Tensor or None]] #

def backward(t: Tensor, grad: Tensor):
    ''''''
    depend_on = t.denpend_on
    t.grad.data += grad.data
    grads = depend_on.grad_fn(grad)
    for i, _grad in enumerate(grads):
        depend_on.inputs[i].backward(_grad)
