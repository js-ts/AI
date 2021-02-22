import _init_path

import pdll as L 
import torch
import numpy as np



data = np.random.rand(10, 10).astype(np.float32)

v = L.autograd.Variable(data[...], requires_grad=True)
t = torch.tensor(data, requires_grad=True)

m_l = L.nn.Linear(10, 20)
m_t = torch.nn.Linear(10, 20)

m_l.weight.tensor = m_t.weight.data.numpy().transpose(1, 0)[...]
m_l.bias.tensor = m_t.bias.data.numpy()[...]

o_l = m_l(v)
o_t = m_t(t)

o_l.mean().backward()
o_t.mean().backward()

# np.testing.assert_almost_equal(o_l.data.numpy(), o_t.data.numpy(), decimal=5)
# np.testing.assert_almost_equal(m_l.weight.grad.numpy(), m_t.weight.grad.numpy().transpose(1, 0), decimal=4)
# np.testing.assert_almost_equal(m_l.bias.grad.numpy(), m_t.bias.grad.numpy(), decimal=4)
# np.testing.assert_almost_equal(v.grad.numpy(), t.grad.numpy())
