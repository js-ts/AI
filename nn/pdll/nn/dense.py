import math
import numpy as np

from .module import Module
from .parameter import Parameter


class Linear(Module):
    """Linear 
    """
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool=True):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        k = math.sqrt(1. / input_dim)
        init_weight = np.random.uniform(low=-k, high=k, size=(input_dim, output_dim))
        self.weight = Parameter(data=init_weight)
        if self.use_bias:
            init_bias = np.random.uniform(low=-k, high=k, size=(output_dim, ))
            self.bias = Parameter(data=init_bias)
        
    def forward(self, data):
        if self.use_bias:
            return data @ self.weight + self.bias
        else:
            return data @ self.weight

    def ext_repr(self, ):
        return f'({self.input_dim}, {self.output_dim})'    
