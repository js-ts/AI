
import _init_path


import pdll as L
from pdll.backend import np 
import torch
import numpy 

data = L.rand(2, 3, requires_grad=True)
mm = L.nn.Linear(3, 10)
out = mm(data)

out.mean().backward()

print(data.grad)
print(data.data, type(data.data))


a = np.random.rand(1, 2, 3)
b = np.random.rand(1, 3)

v_a = L.autograd.Tensor(a[...], requires_grad=True)
v_b = L.autograd.Tensor(b[...], requires_grad=True)
v_c = 2 - 1 - v_a + v_b + 3 -2
v_c = 1 * -v_c * 3 / 3
v_c = 1 / v_c
v_c.backward()

t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, requires_grad=True)
t_c = 2 - 1 - t_a + t_b + 3 -2 
t_c = 1 * -t_c * 3 / 3
t_c = 1 / t_c
t_c.backward(torch.ones_like(t_c))

numpy.testing.assert_almost_equal(v_a.grad.numpy().get(), t_a.grad.data.numpy(), decimal=4)
numpy.testing.assert_almost_equal(v_b.grad.numpy().get(), t_b.grad.data.numpy(), decimal=4)
