from .tensor import Tensor


class ExecutionEngine:

    def run_backward(self, t: Tensor, grad: Tensor):
        raise NotImplementedError

