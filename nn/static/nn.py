
from typing import Sequence, Iterator, Tuple

from .tensor import Tensor
from .layer import Layer 


class Network:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, data: Tensor) -> Tensor:
        for layer in self.layers:
            data = layer.forward(data)
        return data 

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self, ) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for n, param in layer.params.items():
                grad = layer.grads[n]
                yield param, grad