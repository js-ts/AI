import numpy as np 
from typing import Union, List

from autograd.tensor import Tensor


class Parameter(Tensor):

    def __init__(self, *shape) -> None:
        # Importance: uniform [-1, 1]
        data = 2 * np.random.rand(*shape) - 1.
        super().__init__(data, requires_grad=True)
