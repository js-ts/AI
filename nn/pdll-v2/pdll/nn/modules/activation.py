
from pdll.autograd import Variable

from ..functional import relu, sigmoid, tanh
from .module import Module


__all__ = [
    'Tanh', 'Sigmoid', 'ReLU'
]

class Tanh(Module):
    '''tanh
    '''
    def __init__(self, ):
        super().__init__()

    def forward(self, data: Variable) -> Variable:
        return data.tanh()

    def ext_repr(self, ) -> str:
        return ''


class Sigmoid(Module):
    '''sigmoid
    ''' 
    def __init__(self, ):
        super().__init__()

    def forward(self, data: Variable) -> Variable: 
        return data.sigmoid()
    
    def ext_repr(self, ) -> str:
        return ''

    
class ReLU(Module):
    '''relu
    '''
    def __init__(self, ):
        super().__init__()
        
    def forward(self, data: Variable) -> Variable:
        return data.relu()

    def ext_repr(self, ) -> str:
        return ''
