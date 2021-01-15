import numpy as np

from autograd import Tensor


a = Tensor(np.random.rand(3, 3), requires_grad=True)
b = a + 3

c = a - b
d = -c

e = 5 * d

e = e * e 

e.sum().backward()