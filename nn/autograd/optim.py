
from autograd.module import Module

class SGD:
    def __init__(self, module, lr: float=0.01) -> None:
        self.module = module
        self.lr = lr
    
    def step(self) -> None:
        for p in self.module.parameters():
            p -= p.grad * self.lr
    