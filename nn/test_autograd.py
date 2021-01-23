import numpy as np

from autograd import Tensor, Parameter
from autograd import Module, Linear, Tanh


def train():

    data = np.random.rand(1000, 1)
    label = np.cos(data)

    data = Tensor(data)
    label = Tensor(label)

    

if __name__ == "__main__":
    
    train()
    
