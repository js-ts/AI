import numpy as np 
import torch
from autograd import Variable
import autograd

data = np.random.rand(4, 4)

v = Variable(data, requires_grad=True)
v2 = v.reshape(2, 8)
v1 = v2[1:, 2:]
loss = ((v1**2 - v1 + 4) * 3 / 3).exp().mean()
loss.backward()
print(v.grad)

out = 2 ** v1

t = torch.tensor(data, requires_grad=True)
t2 = t.reshape(2, 8)
t1 = t2[1:, 2:]
loss = (3 * (t1**2 - t1 + 4 ) / 3.).exp().mean()
loss.backward()
print(t.grad)


np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)


# ---
linear = autograd.variable.Linear(20, 32)
print(linear.weight.shape)
print(linear.bias.shape)
data = np.random.rand(8, 100, 20)
out = linear(Variable(data))
print(out.shape)
print(linear)


class Model(autograd.variable.Module):
    def __init__(self, ):
        self.l1 = autograd.variable.Linear(10, 20)
        self.l2 = autograd.variable.Linear(20, 30)
        self.conv1 = autograd.variable.Conv2d(3, 8, 3, 2, 1)
    def forward(self, data):
        pass

print(Model())



data = np.random.rand(10, 3, 15, 15)
data = Variable(data, requires_grad=True)

conv = autograd.variable.Conv2d(3, 8, 3, 2, 1)
print(conv)
out = conv(data)
print(out.shape)

loss = out.mean()
loss.backward()

pool = autograd.variable.Pool2d(3, 2, 1, mode='max')
print(pool)
out = pool(data)
print(out.shape)
out.mean().backward()


_data = data.transpose(1, 0, 2, 3)
print(_data.shape)

