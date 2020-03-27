import torch
import torchtext
from torchtext.datasets import text_classification

NGRAMS = 2
BATCH_SIZE = 16
root = '../../data'

import os 
if not os.path.isdir(root):
    os.mkdir(root)

train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=root, ngrams=NGRAMS, vocab=None)


import torch.nn as nn
import torch.nn.functional as F 
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextSentiment, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
    
    def forward(self, text, offset):
        embeded = self.embedding(text, offset)
        return self.fc(embeded)

    def init_weights(self, ):
        x = 0.5
        self.embedding.weight.data.uniform_(-x, x)
        self.fc.weight.data.uniform_(-x, x)
        self.fc.bias.data.zero_()
    

vocab_size = len(train_dataset.get_vocab())
embed_dim = 32
num_class = len(train_dataset.get_labels())
model = TextSentiment(vocab_size, embed_dim, num_class)

print(vocab_size)
print(type(train_dataset.get_vocab()))
print(model)

from torch.utils.data import DataLoader
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=4.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train):
    ''''''
    train_loss = 0
    train_acc = 0
    dataloader = DataLoader(sub_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    for text, offsets, clss in dataloader:
        optimizer.zero_grad()
        output = model(text, offsets)
        loss = criterion(output, clss)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (output.argmax(1) == clss).sum().item()

    scheduler.step()

    return train_loss / len(sub_train), train_acc / len(sub_train)


def test(data):
    total_loss = 0
    acc = 0
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, collate_fn=generate_batch)

    for text, offsets, clss in dataloader:

        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, clss)
            total_loss += loss.item() 
            acc += (output.argmax(1) == clss).sum().item()
        
    return total_loss / len(data), acc / len(data)


import time
from torch.utils.data import dataset
N_EPOCHS = 5
min_valid_loss = float('inf')

train_len = int(len(train_dataset) * 0.95)
sub_train, sub_valid = dataset.random_split(train_dataset, [train_len, len(train_dataset) - train_len])


for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_func(sub_train)
    valid_loss, valid_acc = test(sub_valid)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')



from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    '''
    '''
    # tokenlizer = get_tokenizer('basic_english')
    tokenlizer = get_tokenizer('spacy', language='en')

    with torch.no_grad():
        text = torch.tensor([vocab[token] for token in ngrams_iterator(tokenlizer(text), ngrams)])

        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

vocab = train_dataset.get_vocab()
text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

print(ag_news_label[predict(text_str, model, vocab, 2)])
