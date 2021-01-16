
from .nn import Network

class Optimizer:
    def step(self, net: Network) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, net: Network) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
        