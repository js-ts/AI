import numpy as np 
import torch
from autograd import Variable

data = np.random.rand(3, 3)

v = Variable(data, requires_grad=True)
v1 = v[1:, 2:]
loss = (3 * (v1**2 - v1 + 4)).exp().mean()
loss.backward()
print(v.grad)


t = torch.tensor(data, requires_grad=True)
t1 = t[1:, 2:]
loss = (3 * (t1**2 - t1 + 4)).exp().mean()
loss.backward()
print(t.grad)


np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)