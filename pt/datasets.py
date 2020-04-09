import torch
import torch.utils.data as data
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader

import pandas as pd 

# load_data
# 
# pands
df = pd.read_csv('../nlp/data/train.csv', delimiter=',', ) # header=None, names=['src', 'trg']
print(df.shape)
print(list(df.keys()))

for i in range(df.shape[0]):
    print(df.src.values[i], df.trg.values[i])
    break


# dataset
# 
class CDataset(data.Dataset):
    def __init__(self, ):
        pass
    
    def __len__(self, ):
        pass

    def __getitem__(self, i):
        pass
dataset = CDataset()



# collate_fn
# 
# The input to collate_fn is a list of tensors with the size of batch_size, and the collate_fn function packs them into a mini-batch. 
# Pay attention here and make sure that collate_fn is declared as a top level def. This ensures that the function is available in each worker.
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# DataLoader()