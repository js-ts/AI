import inspect

from .tensor import Tensor, tanh, relu
from .parameter import Parameter

from typing import Iterable

class Module:

    def named_parameters(self, ) -> Iterable[Parameter]:
        prefix = self.__class__.__name__
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield prefix + '.' + name, value
            elif isinstance(value, Module):
                yield from value.named_parameters()

    def parameters(self, ) -> Iterable[Parameter]:
        for _, p in self.named_parameters():
            yield p
    
    def zero_grad(self, ) -> None:
        for p in self.parameters():
            p.zero_grad()
    
    def forward(self, *input, **kwargs):
        raise NotImplementedError

    # __call__ = forward # ERROR
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def __repr__(self, ):
        s = self._class_name
        return s + f'({self.extra_repr()})'
    
    def extra_repr(self, ):
        raise NotImplementedError

    @property
    def _class_name(self, ):
        return self.__class__.__name__


class Linear(Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = Parameter(input_dim, output_dim)
        self.b = Parameter(output_dim)

    def forward(self, data: Tensor) -> Tensor:
        out = data @ self.w + self.b
        return out

    def extra_repr(self, ):
        return f'{self.input_dim}, {self.output_dim}'


class Tanh(Module):

    def forward(self, data: Tensor) -> Tensor:
        return tanh(data)
        
    def extra_repr(self, ):
        return ''


class ReLU(Module):

    def forward(self, data: Tensor) -> Tensor:
        return relu(data)
        
    def extra_repr(self, ):
        return ''