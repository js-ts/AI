
from .module import Module
from .functional import op_pool2d
from ..autograd import Variable


class Pool2d(Module):
    '''pooling
    '''
    def __init__(self, kernel_size: int, stride: int, padding:int, mode: str='max'):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        self.mode = mode

    def forward(self, data: Variable) -> Variable:
        return op_pool2d(self.kernel_size, self.stride, self.padding, self.mode)(data)[0]

    def ext_repr(self, ) -> str:
        return f'(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, mode={self.mode})'

