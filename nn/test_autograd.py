import numpy as np

from autograd import Tensor, Parameter


a = Tensor(np.random.rand(3, 3), requires_grad=True)
b = a + 3

c = a - b
d = -c

e = 5 * d

e = e * e 

e.sum().backward()

m1 = Tensor(np.random.rand(3, 2), requires_grad=True)
m2 = Tensor(np.random.rand(2, 2), requires_grad=True)
m3 = m1 @ m2
m3.sum().backward()

item = m1[:, 0]
print(item)
print(item.shape)


p = Parameter(3)
print(p)