import numpy as np 
import torch
from autograd import Variable

data = np.random.rand(3, 3)

v = Variable(data, requires_grad=True)
loss = (3 * (v**2 - v + 4)).exp().mean()
loss.backward()
print(v.grad)


t = torch.tensor(data, requires_grad=True)
loss = (3 * (t**2 - t + 4)).exp().mean()
loss.backward()
print(t.grad)


np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)