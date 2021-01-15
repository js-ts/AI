import inspect

from autograd import Tensor, Parameter
from typing import Iterable

class Module:
    def parameters(self, ) -> Iterable[Parameter]:
        for _, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
    
    def zero_grad(self, ) -> None:
        for p in self.parameters():
            p.zero_grad()
    
    def forward(self, ) -> 'Tensor':
        raise NotImplementedError
    