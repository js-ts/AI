
from .module import Module
from .functional import op_pool2d
from ..autograd import Variable

from typing import Tuple, Optional


class Pool2d(Module):
    '''pooling
    '''
    def __init__(self, kernel_size: Optional[int or Tuple[int]], stride: Optional[int or Tuple[int]], padding:Optional[int or Tuple[int]], mode: str='max'):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif isinstance(padding, (tuple, list)) and len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
        elif isinstance(padding, (tuple, list)) and len(padding) == 4:
            padding = tuple(padding)
        else:
            raise RuntimeError('not suppot padding format')

        self.padding = padding

        self.mode = mode

    def forward(self, data: Variable) -> Variable:
        return op_pool2d(self.kernel_size, self.stride, self.padding, self.mode)(data)[0]

    def ext_repr(self, ) -> str:
        return f'(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, mode={self.mode})'

