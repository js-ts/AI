from .module import Module
from .functional import op_relu, op_sigmoid, op_tanh
from ..autograd import Variable

class Tanh(Module):
    '''tanh
    '''
    def __init__(self, ):
        pass

    def forward(self, data: Variable) -> Variable:
        return op_tanh()(data)[0]

    def ext_repr(self, ) -> str:
        return ''


class Simoid(Module):
    '''sigmoid
    ''' 
    def forward(self, data: Variable) -> Variable: 
        return op_sigmoid()(data)[0]
    
    def ext_repr(self, ) -> str:
        return ''

    
class ReLU(Module):
    '''relu
    '''
    def forward(self, data: Variable) -> Variable:
        return op_relu()(data)[0]

    def ext_repr(self, ) -> str:
        return ''
