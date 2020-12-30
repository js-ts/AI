import torch
import torch.utils.data as data
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader

import io 
import pandas as pd


# load_data
# 
# pands

def pandas_test():
    df = pd.read_csv('../nlp/data/train.csv', delimiter=',', ) # header=None, names=['src', 'trg']
    print(df.shape, list(df.keys()))
    for i in range(df.shape[0]):
        print(df.src.values[i], df.trg.values[i])
        print(df.__getattr__('src').values[i])
        break

def io_test():
    with io.open('', mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    with io.open('', mode='w', encoding='utf-8') as f:
        f.write('')


# dataset
# 
class CDataset(data.Dataset):
    def __init__(self, ):
        self.size = 20
        self.data = torch.arange(100).reshape(20, 5)

    def __len__(self, ):
        return self.size

    def __getitem__(self, i):
        x = self.data[i]
        return x, {i: x}
        

# Dataloader
# 
# The input to collate_fn is a list of tensors with the size of batch_size, and the collate_fn function packs them into a mini-batch. 
# Pay attention here and make sure that collate_fn is declared as a top level def. This ensures that the function is available in each worker.
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# DataLoader()
def collate_fn(samples):
    from functools import reduce
    assert isinstance(samples, list), ''
    data = [x[0].unsqueeze(0) for x in samples]
    data = torch.cat(data, dim=0)
    othe = reduce(lambda a, b: {**a, **b}, [x[1] for x in samples])
    return data, othe

def dataloader_test():
    dataset = CDataset()
    dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate_fn, shuffle=False) # num_workers=0 # disable multiprocess
    for batch in dataloader:
        print(batch)
        break


# Sampler
# 
def sampler_test():
    BATCH_SIZE = 10
    train_tensor, train_y_tensor = torch.rand(100, 100), torch.rand(100, 1)
    train_dataset = TensorDataset(train_tensor, train_y_tensor)
    
    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(train_dataset)
    ddpsampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

