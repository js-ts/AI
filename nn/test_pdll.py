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



if __name__ == '__main__':
    
    unittest.main(verbosity=1)

