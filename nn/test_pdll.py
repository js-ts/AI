import pdll 
import pdll.autograd
import pdll.nn
import pdll as L 

import torch
import numpy as np
import unittest

# ignoir
class Testing(unittest.TestCase):

    def test_add(self, ):
        '''basic op
        '''
        a = np.random.rand(1, 2, 3)
        b = np.random.rand(1, 3)

        v_a = L.autograd.Variable(a[...], requires_grad=True)
        v_b = L.autograd.Variable(b[...], requires_grad=True)
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

        np.testing.assert_almost_equal(v_a.grad, t_a.grad.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v_b.grad, t_b.grad.data.numpy(), decimal=4)


    def test_matmul(self, ):
        '''mat
        '''
        a = np.random.rand(2, 3)
        b = np.random.rand(3, 4)

        v_a = L.autograd.Variable(a[...], requires_grad=True)
        v_b = L.autograd.Variable(b[...], requires_grad=True)
        v_c = v_a @ v_b
        v_c.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = t_a @ t_b
        t_c.backward(torch.ones_like(t_c))

        np.testing.assert_almost_equal(v_a.grad, t_a.grad.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v_b.grad, t_b.grad.data.numpy(), decimal=4)


    def test_sum(self, ):
        '''stats
        '''
        a = np.random.rand(2, 3, 2) 

        v_a = L.autograd.Variable(a[...], requires_grad=True)
        # v_c = v_a.sum().mean()
        v_c = v_a.var()
        v_c.backward()

        t_a = torch.tensor(a, requires_grad=True)
        # t_c = t_a.sum(dim=1).mean()
        t_c = t_a.var()
        t_c.backward(torch.ones_like(t_c))
        
        np.testing.assert_almost_equal(v_c.data, t_c.data.numpy(), decimal=2)
        np.testing.assert_almost_equal(v_a.grad, t_a.grad.data.numpy(), decimal=2)
        # print(a.var(), v_c.data, t_c.data.numpy())
        # print(v_a.grad)
        # print(t_a.grad)


    def test_shape(self, ):
        a = np.random.rand(2, 3, 2) * 4

        v = L.autograd.Variable(a[...], requires_grad=True)
        v1 = v.reshape(-1, 2).transpose(1, 0)
        v1.mean().backward()

        t = torch.tensor(a, requires_grad=True)
        t1 = t.reshape(-1, 2).transpose(1, 0)
        t1.mean().backward()

        np.testing.assert_almost_equal(v1.data, t1.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad, t.grad.data.numpy(), decimal=4)


    def test_pow(self):

        a = np.random.rand(2, 3, 2) * 2 - 1

        v = L.autograd.Variable(a[...], requires_grad=True)
        v1 = 3 ** (v.exp() ** 2)
        v1.mean().backward()

        t = torch.tensor(a, requires_grad=True)
        t1 = 3 ** (t.exp() ** 2)
        t1.mean().backward()

        np.testing.assert_almost_equal(v1.data, t1.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad, t.grad.data.numpy(), decimal=4)


    def test_activation(self):

        a = np.random.rand(2, 3, 2) * 2 - 1

        v = L.autograd.Variable(a[...], requires_grad=True)
        v1 = L.nn.Tanh()(v)
        v1.mean().backward()

        t = torch.tensor(a, requires_grad=True)
        t1 = torch.nn.Tanh()(t)
        t1.mean().backward()

        np.testing.assert_almost_equal(v1.data, t1.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad, t.grad.data.numpy(), decimal=4)


    def test_linear(self, ):
        data = np.random.rand(10, 10).astype(np.float32)

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        m_l = L.nn.Linear(10, 20)
        m_t = torch.nn.Linear(10, 20)

        m_l.weight.data = m_t.weight.data.numpy().transpose(1, 0)[...]
        m_l.bias.data = m_t.bias.data.numpy()[...]

        o_l = m_l(v)
        o_t = m_t(t)

        o_l.mean().backward()
        o_t.mean().backward()

        np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=5)
        np.testing.assert_almost_equal(m_l.weight.grad, m_t.weight.grad.numpy().transpose(1, 0), decimal=4)
        np.testing.assert_almost_equal(m_l.bias.grad, m_t.bias.grad.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad, t.grad.numpy())


    def test_conv(self):

        data = np.random.rand(8, 3, 100, 100).astype(np.float32) * 2 - 1

        m_l = L.nn.Conv2d(3, 8, 5, 2, 1, )
        m_t = torch.nn.Conv2d(3, 8, 5, 2, 1, )

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        m_l.weight.data[...] = m_t.weight.data.numpy()[...]
        m_l.bias.data[...] = m_t.bias.data.numpy()[...]

        o_l = m_l(v)
        o_t = m_t(t)

        o_l.mean().backward()
        o_t.mean().backward()

        np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=5)
        np.testing.assert_almost_equal(m_l.weight.grad, m_t.weight.grad.numpy(), decimal=4)
        np.testing.assert_almost_equal(m_l.bias.grad, m_t.bias.grad.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)


    def test_max_pool(self):

        data = np.random.rand(8, 3, 100, 100).astype(np.float32)

        m_l = L.nn.Pool2d(3, 2, 1, mode='max')
        m_t = torch.nn.MaxPool2d(3, 2, 1, )

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        o_l = m_l(v)
        o_t = m_t(t)

        o_l.mean().backward()
        o_t.mean().backward()

        np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=5)
        np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)


    def test_avg_pool(self, ):

        data = np.random.rand(8, 3, 100, 100).astype(np.float32)

        m_l = L.nn.Pool2d(3, 2, 1, mode='avg')
        m_t = torch.nn.AvgPool2d(3, 2, 1, )

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        for _ in range(10):
            o_l = m_l(v)
            o_t = m_t(t)

            o_l.mean().backward()
            o_t.mean().backward()

            np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=5)
            np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)


    def test_bn(self, ):

        data = np.random.rand(8, 10, 100, 100).astype(np.float32)

        m_l = L.nn.BatchNorm2d(10)
        m_t = torch.nn.BatchNorm2d(10)

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        m_l.weight.data[...] = m_t.weight.data.numpy()[...]
        m_l.bias.data[...] = m_t.bias.data.numpy()[...]

        for _ in range(10):
            o_l = m_l(v)
            o_t = m_t(t)

            o_l.mean().backward()
            o_t.mean().backward()

            np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=4)
            np.testing.assert_almost_equal(m_l.weight.grad, m_t.weight.grad.numpy(), decimal=4)
            np.testing.assert_almost_equal(m_l.bias.grad, m_t.bias.grad.numpy(), decimal=4)
            np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)

            buffers = list(m_t.buffers())
            np.testing.assert_almost_equal(m_l.running_mean, buffers[0].data.numpy(), decimal=4)
            np.testing.assert_almost_equal(m_l.running_var, buffers[1].data.numpy(), decimal=4)


if __name__ == '__main__':
    
    unittest.main(verbosity=1)

