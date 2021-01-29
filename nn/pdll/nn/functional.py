
import numpy as np 

from ..autograd import Function, Tensor


class op_sigmoid(Function):
    """sigmoid
    """
    def forward(self, t: Tensor):
        self.out = 1. / (1. + np.exp(-t)) 
        return self.out
    
    def backward(self, grad):
        return grad * self.out / (1. - self.out + 1e-10)


class op_relu(Function):
    """relu 
    """ 
    def forward(self, t: Tensor) -> Tensor:
        self.mask = t > 0
        return t * self.mask
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.mask


class op_tanh(Function):
    """
    formul: (exp(x) + exp(-x)) / (exp(x) - exp(-x))
    derive : 1 - tanh(x) ** 2
    """
    def forward(self, t: Tensor) -> Tensor:
        self.out = np.tanh(t)
        return self.out
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad  * (1 - self.out ** 2)
