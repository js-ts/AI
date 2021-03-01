import _init_path

import pdll 
import pdll.autograd
import pdll.nn
import pdll as L 

import torch
import numpy as np
import unittest
import copy
import os 

class Testing(unittest.TestCase):

    def test_multihead(self, ):
        ''' '''
        
        data = np.random.rand(8, 3, 16)

        dt = torch.tensor(data, dtype=torch.float32)
        mt = torch.nn.MultiheadAttention(16, 4)

        outt, attt = mt(dt, dt, dt)
        print(attt.shape    )



        dl = L.from_numpy(data[...])
        ml = L.nn.MultiHeadAttention(16, 4)

        ml.in_proj_weight.storage[...] = mt.in_proj_weight.data.numpy()
        ml.in_proj_bias.storage[...] = mt.in_proj_bias.data.numpy()
        ml.out_proj.weight.storage[...] = mt.out_proj.weight.data.t().numpy()
        ml.out_proj.bias.storage[...] = mt.out_proj.bias.data.numpy()

        outl, _ = ml(dl, dl, dl)

        print(ml.in_proj_weight.shape, mt.in_proj_weight.shape)
        print('output', outl.shape)

        np.testing.assert_almost_equal(outl.data.storage, outt.data.numpy(), decimal=4)
        # np.testing.assert_almost_equal(attl.data.storage, attt.data.numpy(), decimal=4)



if __name__ == '__main__':
    
    unittest.main(verbosity=1)

