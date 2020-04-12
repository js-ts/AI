# reference：
# https://arxiv.org/pdf/1706.03762.pdf
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/jadore801120/attention-is-all-you-need-pytorch
# 
# The Annotated Transformer: 
# http://nlp.seas.harvard.edu/2018/04/03/attention.html

import torch 
import torch.nn as nn
from torch import Tensor
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import math
import copy
import matplotlib.pyplot as plt 
import time

"""

de = 'de_core_news_sm'
en = 'en_core_web_sm'
SRC = Field(tokenize='spacy', tokenizer_language=de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize='spacy', tokenizer_language=en, init_token='<sos>', eos_token='<eos>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG), root='../../data')

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
for i in range(6):
    print(i, SRC.vocab.itos[i])

print(len(SRC.vocab), len(TRG.vocab))

batch_size = 3
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size,)


class PositionEncoding(nn.Module):
    def __init__(self, d_hid, n_postion=200):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_postion, d_hid))

    def _get_sinusoid_encoding_table(self, n_postion, d_hid):
        '''
        '''
        def get_position_angle_vec(position):
            return [position / math.pow(10000, 2 * (i // 2) / d_hid) for i in range(d_hid)]
        table = np.array([get_position_angle_vec(pos) for pos in range(n_postion)])
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return torch.tensor(table).unsqueeze(1).to(torch.float)

    def forward(self, x: Tensor):
        '''x: [len, bz, dim]
        '''
        return x + self.pos_table[:x.size(0)].clone().detach()


class Encoder(nn.Module):
    def __init__(self, n_src_vocab: int, d_word_vec: int, pad_idx: int, n_layer: int, n_head: int, d_model: int, dropout: float=0.1, n_postion: int=200):
        super().__init__()
        self.pad_idx = pad_idx
        self.n_layer = n_layer
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionEncoding(d_word_vec, n_postion)
        self.dropout = nn.Dropout(p=dropout)
        self.attenlayers = nn.ModuleList([nn.MultiheadAttention(d_model, n_head, dropout=dropout) for _ in range(n_layer)])
        self.feedforward = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layer)])
        self.layers_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer + n_layer)])

    def forward(self, src):
        '''[len, sz]
        '''
        src_mask = (src == self.pad_idx).permute(1, 0) # [len, bz] -> [bz, len]
        src = self.src_word_emb(src)
        src = self.position_enc(src)
        encode_output = self.dropout(src)

        for i in range(self.n_layer):
            _encode_output, _ = self.attenlayers[i](encode_output, encode_output, encode_output, key_padding_mask=src_mask)
            encode_output += _encode_output
            encode_output = self.layers_norm[i](encode_output)

            _encode_output = self.feedforward[i](encode_output)
            encode_output += _encode_output
            encode_output = self.layers_norm[i+1](encode_output)

        return encode_output



class Decoder(nn.Module):
    def __init__(self, n_trg_vocab: int, d_word_vec: int, pad_idx: int, n_layer: int, n_head: int, d_model: int, dropout: float=0.1, n_postion: int=200):
        super().__init__()
        self.pad_idx = pad_idx
        self.n_layer = n_layer
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionEncoding(d_word_vec, n_postion)
        self.dropout = nn.Dropout(p=dropout)
        self.selfattenlayers = nn.ModuleList([nn.MultiheadAttention(d_model, n_head, dropout=dropout) for _ in range(n_layer)])
        self.decattenlayers = nn.ModuleList([nn.MultiheadAttention(d_model, n_head, dropout=dropout) for _ in range(n_layer)])
        self.feedforward = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layer)])
        self.layers_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer + n_layer + n_layer)])

    def forward(self, trg, enc_output):
        '''[len, sz]
        '''
        trg_mask = (trg == self.pad_idx).permute(1, 0) # [len, bz] -> [bz, len]
        atn_mask = torch.triu(torch.ones((trg.size(0), trg.size(0)))).permute(1, 0)
        atn_mask = atn_mask.masked_fill(atn_mask==1, 0).masked_fill(atn_mask==0, float('-inf'))

        trg = self.trg_word_emb(trg)
        trg = self.position_enc(trg)
        decode_output = self.dropout(trg)

        for i in range(self.n_layer):
            _decode_output, _ = self.selfattenlayers[i](decode_output, decode_output, decode_output, key_padding_mask=trg_mask, attn_mask=atn_mask)
            decode_output += _decode_output
            decode_output = self.layers_norm[i](decode_output)

            _decode_output, _ = self.decattenlayers[i](decode_output, enc_output, enc_output)
            decode_output += _decode_output
            decode_output = self.layers_norm[i+1](decode_output)

            _decode_output = self.feedforward[i](decode_output)
            decode_output += _decode_output
            decode_output = self.layers_norm[i+1](decode_output)

        return decode_output


class Transformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, d_model: int, n_trg_word: int):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.word_proj = nn.Linear(d_model, n_trg_word)

    def forward(self, src_seq, trg_seq):
        enc_output = self.encoder(src_seq)
        dec_output = self.decoder(trg_seq, enc_output)
        logits = self.word_proj(dec_output)

        return logits


if __name__ == '__main__':

    encoder = Encoder(n_src_vocab=len(SRC.vocab), d_word_vec=256, pad_idx=SRC.vocab.stoi['<pad>'], n_layer=5, n_head=4, d_model=256)
    decoder = Decoder(n_trg_vocab=len(TRG.vocab), d_word_vec=256, pad_idx=TRG.vocab.stoi['<pad>'], n_layer=5, n_head=4, d_model=256)
    transforer = Transformer(encoder, decoder, 256, len(TRG.vocab))
    # print(encoder)

    for batch in train_iterator:
        out = transforer(batch.src, batch.trg)

        print(out.shape)

        break

"""

class EncoderDecoder(nn.Module):
    '''
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        '''
        '''
        mem = self.encode(src, src_mask)
        out = self.decode(mem, src_mask, tgt, tgt_mask)
        return out

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, mem, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), mem, src_mask, tgt_mask)
    

class Generator(nn.Module):
    '''linea + softmax
    '''
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


def clones(module, n):
    '''
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    '''
    '''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    '''stack of N layers
    '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        '''
        '''
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    '''
    '''
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    '''
    '''
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        '''
        '''
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    '''
    '''
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mem, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, mem, src_mask, trg_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    '''
    '''
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, mem, src_mask, trg_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, mem, mem, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    '''
    '''
    att_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(att_shape, dtype=np.uint8), k=1)
    return torch.tensor(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    '''dot product attention, [N, Head, Lx, Dim]
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    
    p_attn = torch.softmax(scores, dim=-1) # [N, Head, Lq, Lk]
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h 
        self.h = h 
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).permute(0, 2, 1, 3) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.permute(0, 2, 1, 3).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    '''
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # self.w1 = nn.Linear(d_model, d_ff)
        # self.w2 = nn.Linear(d_ff, d_model)
        self.linear = nn.Sequential(nn.Linear(d_model, d_ff),
                                     nn.ReLU(), 
                                     nn.Dropout(dropout), 
                                     nn.Linear(d_ff, d_model))
    def forward(self, x):
        return self.linear(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        po = torch.arange(0, max_len).unsqueeze(1)
        div_iterm = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(po * div_iterm)
        pe[:, 1::2] = torch.cos(po * div_iterm)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

def make_model(src_vocab, trg_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    '''
    '''
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, trg_vocab), c(position)),
        Generator(d_model, trg_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model

tmp_model = make_model(10, 10, 2)
print(tmp_model)


class NoamOpt(object):
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    def step(self, ):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

# Three settings of the lrate hyperparameters.
# opts = [NoamOpt(512, 1, 4000, None), 
#         NoamOpt(512, 1, 8000, None),
#         NoamOpt(256, 1, 4000, None)]
# plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
# plt.legend(["512:4000", "512:8000", "256:4000"])
# plt.show()

class LabelSmoothing(nn.Module):
    '''
    '''
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size 
        self.true_dist = None
    
    def forward(self, x, target):
        # x[-1, size], target: [-1]
        assert x.size(1) == self.size 
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.detach())

# crit = LabelSmoothing(5, 0, 0.4)
# predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0], 
#                              [0, 0.2, 0.7, 0.1, 0]])
# v = crit(predict.log(), torch.LongTensor([2, 1, 0]))
# plt.imshow(crit.true_dist)
# plt.show()

# crit = LabelSmoothing(5, 0, 0.1)
# def loss(x):
#     d = x + 3 * 1
#     predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],])
#     return crit(predict.log(), torch.LongTensor([1]))
# plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
# plt.show()


class Batch(object):
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # [batch, 1, Ls]
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.mask_std_mask(self.trg, pad) # [batch, Lt, Lt]
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def mask_std_mask(trg, pad):
        trg_mask = (trg != pad).unsqueeze(-2)
        trg_mask = trg_mask & subsequent_mask(trg.size(-1)).to(dtype=trg_mask.dtype)
        return trg_mask


def data_gen(V, batch, nbatches):
    '''
    '''
    for _ in range(nbatches):
        data = torch.tensor(np.random.randint(0, V, size=(batch, 10))).to(dtype=torch.long)
        data[:, 0] = 1
        yield Batch(data, copy.deepcopy(data), 0)


class SimpleLossCompute(object):
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.item() * norm


def run_epoch(data_iter, model, loss_compute):
    '''
    '''
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %(i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    
    return total_loss / total_tokens


v = 11
criterion = LabelSmoothing(size=v, padding_idx=0, smoothing=0.)
model = make_model(v, v, N=2)
model_opt = torch.optim.Adam(model.parameters(), lr=0., betas=(0.9, 0.98), eps=1e-9)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, model_opt)


for epoch in range(10):
    model.train()
    run_epoch(data_gen(v, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    run_epoch(data_gen(v, 30, 20), model, SimpleLossCompute(model.generator, criterion, None))
