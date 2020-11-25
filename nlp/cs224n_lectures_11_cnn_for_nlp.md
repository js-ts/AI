# conv for nlp

- classification
    - word level
    - character level

- translation
    - cnn for encoder
    - rnn for decoder

- gated units used vertivally
    - residual
    - highway

- regulation
    - bn
    - dropout

- pooling
    - maxpool
    - kmaxpool

- Q-RNN
    - cnn
    - fast

- convolutional seq2seq

```
import torch
import torch.nn as nn

batch = 8
length = 7 # shape [same as HW]
feadim = 100 # input channel 

# conv
data = torch.rand(batch, feadim, length)
m = nn.Conv1d(feadim, out_channels=32, kernel_size=3, stride=2, padding=1)
out = m(data)
print(out.shape)

# maxpool
pool, _ = out.max(dim=-1)
print(pool.shape)

# kmaxpool
def kmax_pooling(x, dim, k):
    idx = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, idx)

v = kmax_pooling(out, dim=-1, k=2)
print(v.shape)

```