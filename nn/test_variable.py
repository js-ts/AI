import numpy as np 
import torch
from autograd import Variable

data = np.random.rand(4, 4)

v = Variable(data, requires_grad=True)
v2 = v.reshape(2, 8)
v1 = v2[1:, 2:]
loss = ((v1**2 - v1 + 4) * 3 / 3).exp().mean()
loss.backward()
print(v.grad)


t = torch.tensor(data, requires_grad=True)
t2 = t.reshape(2, 8)
t1 = t2[1:, 2:]
loss = (3 * (t1**2 - t1 + 4 ) / 3.).exp().mean()
loss.backward()
print(t.grad)


np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)