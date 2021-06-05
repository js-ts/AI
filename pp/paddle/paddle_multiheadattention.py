import torch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F    



# https://github.com/PaddlePaddle/Paddle/issues/33124

data = [torch.rand(1, 3, 9, ) for _ in range(3)]
mm = torch.nn.MultiheadAttention(9, 3, )
out, weight = mm(*data)

print(mm.in_proj_weight.shape)
print(mm.in_proj_bias.shape)
print(mm.out_proj.weight.shape)
print(mm.out_proj.bias.shape)


# data format 
pdata = [paddle.to_tensor(d.numpy().transpose(1, 0, 2)) for d in data]

pmm = paddle.nn.MultiHeadAttention(9, 3, need_weights=True)

pmm.q_proj.weight.set_value( mm.in_proj_weight[:9].permute(1, 0).data.numpy() )
pmm.q_proj.bias.set_value( mm.in_proj_bias.data.numpy()[:9] )

pmm.k_proj.weight.set_value( mm.in_proj_weight[9:9+9].permute(1, 0).data.numpy() )
pmm.k_proj.bias.set_value( mm.in_proj_bias.data.numpy()[9:9+9] )

pmm.v_proj.weight.set_value( mm.in_proj_weight[9+9:].permute(1, 0).data.numpy() )
pmm.v_proj.bias.set_value( mm.in_proj_bias.data.numpy()[9+9:] )

pmm.out_proj.weight.set_value( mm.out_proj.weight.permute(1, 0).data.numpy() )
pmm.out_proj.bias.set_value( mm.out_proj.bias.data.numpy() )

pout, pweight = pmm(*pdata)

print(out.mean())
print(pout.mean())
