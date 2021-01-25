import numpy as np

# from .tensor import Tensor
# from .tensor import Dependency

from typing import Any
from collections import OrderedDict


class _ContextMixin(object):

    def save_for_backward(self, *tensors):
        self.to_save = tensors
    

class _BaseFunction(object):
    pass



class _Function(_BaseFunction, _ContextMixin):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
