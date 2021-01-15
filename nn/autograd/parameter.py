import numpy as np 
from typing import Union, List

from autograd.tensor import Tensor


class Parameter(Tensor):

    def __init__(self, *shape) -> None:

        data = np.random.rand(*shape)
        print(data.shape)
        print(data)
        super().__init__(data, requires_grad=True)
