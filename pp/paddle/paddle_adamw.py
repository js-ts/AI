import torch
import paddle

import numpy as np



def get_torch_mm():
    m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, 2, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Flatten(),
            torch.nn.Linear(50 * 50 * 8, 10),
            torch.nn.ReLU()
        )
    
    return m


def get_paddle_mm():
    m = paddle.nn.Sequential(
            paddle.nn.Conv2D(3, 8, 3, 2, 1),
            paddle.nn.BatchNorm2D(8),
            paddle.nn.Flatten(),
            paddle.nn.Linear(50 * 50 * 8, 10),
            paddle.nn.ReLU()
        )
            
    return m


def reset_parameters(paddle_model, torch_model, data=None):
    '''reset_parameters
    '''
    sublayers = [(n, m) for n, m in paddle_model.named_sublayers(include_self=True)]
    submoduls = [(n, m) for n, m in torch_model.named_modules()]
    
    print(len(sublayers), len(submoduls))
    assert len(sublayers) == len(submoduls), f'{len(sublayers)}, {len(submoduls)}'
    
    for i, (n, m) in enumerate(sublayers):

        _m = submoduls[i][1]
        
        if isinstance(m, paddle.nn.Conv2D):
            m.weight.set_value( _m.weight.data.numpy() )
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                m.bias.set_value( _m.bias.data.numpy() )
                
        elif isinstance(m, paddle.nn.Linear):
            m.weight.set_value( _m.weight.data.numpy().T )
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                m.bias.set_value( _m.bias.data.numpy() )
        
        elif isinstance(m, paddle.nn.BatchNorm2D):
            m.weight.set_value( _m.weight.data.numpy() )
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                m.bias.set_value( _m.bias.data.numpy() )
            
            m._mean.set_value( _m.running_mean.data.numpy() )
            m._variance.set_value( _m.running_var.data.numpy() )
    
    
    if data is None:
        data = np.random.rand(1, 3, 100, 100).astype(np.float32)

    tdata = torch.tensor(data[...])
    pdata = paddle.to_tensor(data[...])
    tdata.requires_grad = True
    pdata.stop_gradient = False

    tout = torch_model(tdata)
    pout = paddle_model(pdata)

    print('-------forward---------')
    print('torch', tout.mean(), tout.sum())
    print('paddle', pout.mean(), pout.sum())
    np.testing.assert_almost_equal(pout.numpy(), tout.data.numpy(), decimal=4)

    print('-------backward---------')
    tout.sum().backward()
    pout.sum().backward()

    print('torch', tdata.grad.mean(), tdata.grad.sum())
    print('paddle', pdata.grad.mean(), pdata.grad.sum())
    np.testing.assert_almost_equal(pdata.grad.numpy(), tdata.grad.data.numpy(), decimal=4)

    _check_parameters(paddle_model, torch_model)
    
    

def _check_parameters(pm, tm):
    '''
    '''
    tp = [p for p in tm.parameters() if p.requires_grad]
    pp = [p for p in pm.parameters() if not p.stop_gradient]
    assert len(tp) == len(pp), ''
    
    for _tp, _pp in zip(tp, pp):
        try:
            np.testing.assert_almost_equal(_tp.data.numpy(), _pp.numpy(), decimal=4)
            np.testing.assert_almost_equal(_tp.grad.data.numpy(), _pp.grad.numpy(), decimal=4)
        except:
            np.testing.assert_almost_equal(_tp.data.numpy().T, _pp.numpy(), decimal=4)
            np.testing.assert_almost_equal(_tp.grad.data.numpy().T, _pp.grad.numpy(), decimal=4)


def check_optimizer(optim):
    
    for e in range(10):
        for _ in range(10):
            data = np.random.rand(1, 3, 100, 100).astype(np.float32)
            tdata = torch.tensor(data[...])
            pdata = paddle.to_tensor(data[...])

            

tm = get_torch_mm()
pm = get_paddle_mm()

reset_parameters(pm, tm)

