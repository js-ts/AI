from typing import Tuple

from pdll.autograd import Variable
from pdll.backend import np

class Parameter(Variable):
    '''parameter
    '''
    def __init__(self, *shape, data=None):
        if shape:
            data = np.random.rand(shape) * 2 - 1
        super().__init__(data, requires_grad=True)
        self.grad = np.zeros_like(self.data)
