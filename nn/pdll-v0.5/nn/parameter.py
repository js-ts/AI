import numpy as np

from ..autograd import Variable


class Parameter(Variable):

    def __init__(self, *shape, data=None):
        if shape:
            data = np.random.rand(shape) * 2 - 1
        super().__init__(data, requires_grad=True)

