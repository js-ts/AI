import numpy as np 
from typing import Union, List

from .tensor import Tensor


class Parameter(Tensor):

    def __init__(self, shape: Union[list, tuple, int]) -> None:
        if isinstance(shape, int):
            shape = (shape, )
        data = np.random.rand(*shape)
        super().__init__(data=data, requires_grad=True)
