import math
import numpy as np 

from .parameter import Parameter
from .module import Module
from .functional import op_conv2d
from ..autograd import Variable


class Conv2d(Module):
    '''
    image: C_in H_in W_in
    kernel: C_out C_in H_kernel W_kernel
    output: C_out H_out W_out
    H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1)
    '''
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int=1, groups: int=1, bias: bool=True):
        self.in_channels = in_channels
        self.out_channels = out_channels

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

        self.dilation = dilation
        self.groups = groups
        
        assert in_channels % groups == 0, 'k * 1/k * 1/k'

        k = math.sqrt(1. / (groups * in_channels * kernel_size[0] * kernel_size[1]))
        weight_init = np.random.uniform(low=-k, high=k, size=(out_channels, int(in_channels/groups), kernel_size[0], kernel_size[1]))
        self.weight = Parameter(data=weight_init)
        
        if bias:
            bias_init = np.random.uniform(low=-k, high=k, size=(self.out_channels, ))
            self.bias = Parameter(data=bias_init)
        else:
            self.bias = None

    def forward(self, data: Variable) -> Variable:
        return op_conv2d(self.kernel_size, self.stride, self.padding, self.dilation)(data, self.weight, self.bias)[0]


    def ext_repr(self, ) -> str:
        s = f'({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'
        if self.dilation != 1:
            s += f'dilation={self.dilation}'
        if self.groups != 1:
            s += f'groups={self.groups}'

        return s
