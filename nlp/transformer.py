# reference：
# https://arxiv.org/pdf/1706.03762.pdf
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/jadore801120/attention-is-all-you-need-pytorch
# 

import torch 
import torch.nn as nn
from torch import Tensor
import numpy as np
import math
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

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