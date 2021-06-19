import torch
import paddle


mm = torch.nn.Linear(5, 5)
pm = paddle.nn.Linear(5, 5)

pm.weight.set_value( mm.weight.data.numpy().T )
if hasattr(mm, 'bias') and getattr(mm, 'bias') is not None:
    pm.bias.set_value( mm.bias.data.numpy() )


data = torch.rand(3, 5)
mm(data).sum().backward()
pm(paddle.to_tensor(data.data.numpy())).sum().backward()


print(mm.weight.grad.data.numpy())
print(pm.weight.grad.numpy().T)
print()

tnorm = torch.nn.utils.clip_grad_norm_(mm.parameters(), 0.1)
print(mm.weight.grad.data.numpy())

pclip = paddle.nn.ClipGradByGlobalNorm(0.1)
params_grads = [(p, p.grad) for p in pm.parameters()]
params_grads = pclip._dygraph_clip(params_grads)
print(params_grads[0][1].numpy().T)
