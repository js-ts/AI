import torch 
import torch.nn as nn

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

de = 'de_core_news_sm'
en = 'en_core_web_sm'
SRC = Field(tokenize='spacy', tokenizer_language=de, init_token='<SOS>', eos_token='<EOS>', lower=True)
TRG = Field(tokenize='spacy', tokenizer_language=en, init_token='<SOS>', eos_token='<EOS>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG), root='../../data')

print(train_data[0].__dict__['src'])
print(train_data[0].__dict__['trg'])

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
for i in range(4):
    print(SRC.vocab.itos[i])
for i in range(4):
    print(TRG.vocab.itos[i])   

batch_size = 3
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size,)

for batch in train_iterator:
    print(batch.src.shape)
    print(batch.trg.shape)
    break

import random
import time
import math
from typing import Tuple
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        embded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super(Attention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor):
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
        attention = torch.sum(energy, dim=2)

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float, attention: nn.Module):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(enc_hid_dim * 2 + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self, input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor]:
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
        rnn_input = torch.cat((weighted_encoder_rep, embedded), dim=2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))

        return output, decoder_hidden.squeeze(0)

    
class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float=0.5) -> Tensor:
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        output = trg[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = trg[t] if teacher_forcing else top1
        
        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

device = torch.device('cpu')

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)
optimizer = optim.Adam(model.parameters())

PAD_IDX = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

print(model)



def train(model: nn.Module, iterator: BucketIterator, optimizer: optim.Optimizer, criterion: nn.Module, clip: float=1):
    ''' '''
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print(i, loss.item())

    return epoch_loss / len(iterator)



def evaluate(model: nn.Module, iterator: BucketIterator, criterion: nn.Module):
    ''' '''
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0) #turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


for epoch in range(1):
    train_loss = train(model, train_iterator, optimizer, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)
    print(train_loss)

