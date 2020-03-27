# referrence: 
# 
# https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
# https://github.com/pytorch/text/issues/78
# 
# Field
# Example
# Dataset: 
#   building exmaples (in Example will use field preprocess)
# Iterator: 
#   using Dataset to build Batch

import torch 
import torch.nn as nn
import torchtext
from torchtext.data import Field
from torchtext.data import Pipeline
from torchtext.data import TabularDataset # for csv/csv
from torchtext.data import Iterator, BucketIterator
from torchtext.data import BPTTIterator
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import Vocab, build_vocab_from_iterator

import numpy as np
from collections import Counter

import spacy
from spacy.symbols import ORTH

# Tokenize
nlp = spacy.load("en_core_web_sm")
nlp.tokenizer.add_special_case("don't", [{ORTH: "do"}, {ORTH: "not"}])
def tokenize(s):
    return [t.text for t in nlp.tokenizer(s)]

# tokenize = lambda s : s.strip().split(' ')


# Field  
def PreProcess(s):
    return float(s)
TEXT = Field(sequential=True, tokenize=tokenize, lower=True, init_token='<sos>', eos_token='<eos>')
LABEL = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=Pipeline(PreProcess))

# Dataset, tsv, csv
# # when building dataset, Feild's <preprocessing> will run when create exmaple func. 
# # filter_pred

def filter_pred(example):
    return example.__dict__['src'] is not None

datafields = [('id', None), ('src', TEXT), ('trg', TEXT), ('label', LABEL)]
train, valid = TabularDataset.splits(path='./data', fields=datafields, train='train.csv', validation='valid.csv', format='csv', skip_header=True)

datafields = [('id', None), ('src', TEXT), ('trg', TEXT)]
test = TabularDataset(path='data/test.csv', fields=datafields, format='csv', skip_header=True, filter_pred=filter_pred)

# Example
print(train[0].__dict__.keys())# data.exmaple.Example
print(train[0].__dict__['src'])

# Vocab
# TEXT.build_vocab(train, min_freq=2, vectors="glove.6B.200d")
TEXT.build_vocab(train, min_freq=1,)

# Word Hash Vocab
letts_counter = Counter()
specials = ['<unk>', '<pad>', '<sos>', '<eos>']
letter_n_gram = 3
for i in range(len(TEXT.vocab)):
    w = TEXT.vocab.itos[i]
    if w in specials:
        continue
    w = '#' + w + '#'
    c = zip(*[list(w)[i:] for i in range(letter_n_gram)])
    c = [''.join(x) for x in c]
    letts_counter.update(c)

letter_vocab = Vocab(letts_counter, specials=specials)

vectors = [0 for _ in range(len(TEXT.vocab))]
stoi = {}
for i in range(len(TEXT.vocab)):
    w = TEXT.vocab.itos[i]
    if w not in specials:
        c = zip(*[list('#'+w+'#')[i:] for i in range(letter_n_gram)])
        c = [''.join(x) for x in c]
    else:
        # c = [w]
        continue
    _idx = [letter_vocab.stoi[x] for x in c]
    _dim = len(letter_vocab)
    _vec = np.eye(_dim)[_idx, :].sum(axis=0)
    vectors[i] = torch.tensor(_vec).to(torch.float) # 
    stoi[w] = i

TEXT.vocab.set_vectors(stoi, vectors, _dim, unk_init=torch.Tensor.zero_)


# Iterator
# will create and reture Batch object, 
# will call <process> for each filed before return in Batch Object
bs = 2
train_iter, valid_iter = BucketIterator.splits((train, valid), batch_sizes=(bs, bs), sort_key=lambda x: len(x.src))
test_iter = Iterator(test, batch_size=bs, )

# BPTTIterator
# BPTTIterator

weight = TEXT.vocab.vectors
print(weight.shape)

dim = len(letter_vocab)
embedding = nn.Embedding(len(TEXT.vocab), dim)
embedding.weight.data.copy_(TEXT.vocab.vectors)

for batch in train_iter:
    src = batch.src
    trg = batch.trg
    lab = batch.label
    print('src1: ', src.shape) # [lens, batch, dims]

    src = embedding(src)
    lens, bss, dim = src.shape
    print('src2: ', src.shape)

    word_n_gram = 3
    src = [torch.cat(t, dim=-1) for t in zip(*[src[i:] for i in range(word_n_gram)])]
    src = torch.cat(src, dim=0).view(-1, bss, word_n_gram * dim)
    print('src3: ', src.shape)

