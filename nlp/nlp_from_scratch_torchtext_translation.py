import torch 
import torch.nn as nn

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

de = 'de_core_news_sm'
en = 'en_core_web_sm'
SRC = Field(tokenize='spacy', tokenizer_language=de, init_token='<SOS>', eos_token='<EOS>', lower=True)
TRG = Field(tokenize='spacy', tokenizer_language=en, init_token='<SOS>', eos_token='<EOS>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG), root='../../data')

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

batch_size = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size,)

