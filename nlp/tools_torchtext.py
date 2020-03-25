# referrence: 
# 
# https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
# 

import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset # for csv/csv
from torchtext.data import Iterator, BucketIterator
from torchtext.data import BPTTIterator

import spacy
from spacy.symbols import ORTH

# Tokenize
nlp = spacy.load("en_core_web_sm")
nlp.tokenizer.add_special_case("don't", [{ORTH: "do"}, {ORTH: "not"}])
def tokenize(s):
    return [t.text for t in nlp.tokenizer(s)]

# tokenize = lambda s : s.strip().split(' ')

# Field
TEXT = Field(sequential=True, tokenize=tokenize, lower=True, init_token='<sos>', eos_token='<eos>')
LABEL = Field(sequential=False, use_vocab=False, lower=True)

# Dataset, tsv, csv
datafields = [('id', None), ('src', TEXT), ('trg', TEXT), ('label', LABEL)]
train, valid = TabularDataset.splits(path='./data', fields=datafields, train='train.csv', validation='valid.csv', format='csv', skip_header=True)

datafields = [('id', None), ('src', TEXT), ('trg', TEXT)]
test = TabularDataset(path='data/test.csv', fields=datafields, format='csv', skip_header=True)

# Example
print(train[0].__dict__.keys())# data.exmaple.Example
print(train[0].__dict__['src'])

# Vocab
# TEXT.build_vocab(train, min_freq=2, vectors="glove.6B.200d")
TEXT.build_vocab(train, min_freq=1,)

for i in range(len(TEXT.vocab)):
    print(i, TEXT.vocab.itos[i])

# Iterator
train_iter, valid_iter = BucketIterator.splits((train, valid), batch_sizes=(3, 3), sort_key=lambda x: len(x.src))
test_iter = Iterator(test, batch_size=2, )

# BPTTIterator
# BPTTIterator


for batch in train_iter:
    src = batch.src
    trg = batch.trg
    lab = batch.label
    print(src)
    print(trg)
    print(lab)
    print(src.shape)
    break


# for batch in valid_iter:
#     src = batch.src
#     trg = batch.trg 
#     lab = batch.label
#     print(src)
#     print(trg)
#     print(lab)
#     break