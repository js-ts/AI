
from pdll.autograd import Variable

from ..functional import softmax 
from ..functional import cross_entropy

from .module import Module


class Softmax(Module):
    '''softmax
    ''' 
    def __init__(self, axis: int=-1):
        super().__init__()
        self.axis = axis

    def forward(self, data: Variable) -> Variable: 
        return softmax(data, self.axis)
    
    def ext_repr(self, ) -> str:
        return ''



class CrossEntropyLoss(Module):
    '''crossentropyloss
    ''' 
    def __init__(self, axis: int=-1, reduction: str='mean'):
        super().__init__()
        self.axis = axis
        self.reduction = reduction

    def forward(self, data: Variable, label: Variable) -> Variable: 
        return cross_entropy(data, label, axis=self.axis, reduction=self.reduction)

    def ext_repr(self, ) -> str:
        return ''
