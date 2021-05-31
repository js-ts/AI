import paddle
import torch

print('paddle.__version__', f'{paddle.__version__}')
print('torch.__version__', f'{torch.__version__}')


pm = paddle.nn.Conv2D(3, 8, 3, 2, 1)
data = paddle.rand([1, 3, 10, 10])

out = pm(data)
out.sum().backward()

print('paddle before clip, pm.weight.grad', pm.weight.grad.sum(),  pm.weight.grad.mean(), pm.weight.grad.norm())
print('paddle before clip, pm.bias.grad', pm.bias.grad.sum(),  pm.bias.grad.mean(), pm.bias.grad.norm())

pclip = paddle.nn.ClipGradByGlobalNorm(0.1)
out = pclip( [(pm.weight, pm.weight.grad), (pm.bias, pm.bias.grad)] )

print('paddle after clip pm.weight.grad', out[0][1].sum(), out[0][1].mean(), out[0][1].norm())
print('paddle after clip pm.bias.grad', out[1][1].sum(), out[1][1].mean(), out[1][1].norm())

# ------torch

tm = torch.nn.Conv2d(3, 8, 3, 2, 1)
tdata = torch.tensor(data.numpy())

tm.weight.data = torch.tensor(pm.weight.numpy())
tm.bias.data = torch.tensor(pm.bias.numpy())

tout = tm(tdata)
tout.sum().backward()

print('torch before clip, tm.weight.grad', tm.weight.grad.sum(), tm.weight.grad.mean(), tm.weight.grad.norm())
print('torch before clip, tm.bias.grad', tm.bias.grad.sum(), tm.bias.grad.mean(), tm.bias.grad.norm())

torch.nn.utils.clip_grad_norm_(tm.parameters(), 0.1)

print('torch after clip, tm.weight.grad', tm.weight.grad.sum(), tm.weight.grad.mean(), tm.weight.grad.norm())
print('torch after clip, tm.bias.grad', tm.bias.grad.sum(), tm.bias.grad.mean(), tm.bias.grad.norm())
