from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn 
import numpy as np 
import glob
import random
import string
import unicodedata
import io
import os
import time 
import math 


def findFiles(path):
    return glob.glob(path)

root_path = '../../../Downloads/data/data/names/*.txt'
# print(findFiles('../../../Downloads/data/data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

print(n_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))


def readLines(filename):
    lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]



all_categories = []
category_lines = {}

for filename in findFiles(root_path):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# print(n_categories)
# print(category_lines['Italian'][:5])

# print('sdasdsa'.find('a'))
# print('sdasfsd'.index('a'))


def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    '''
    '''
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    '''
    '''
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, c in enumerate(line):
        tensor[i][0][letterToIndex(c)] = 1

    return tensor


print(letterToTensor('J').size())
print(lineToTensor('Jones').size())



# -----------------

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        '''
        '''
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self, ):
        return torch.zeros(1, self.hidden_size)


device = 'cuda' if not torch.cuda.is_available() else 'cpu'

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# print(rnn)

# inputx = lineToTensor('Albert')
# hidden = torch.zeros(1, n_hidden)
# print(inputx.shape)

# output, next_hidden = rnn(inputx[0], hidden)
# print(output.shape)
# print(output.topk(2))

def categoryFromOutput(out):
    _, idx = output.topk(1)
    c_i = idx[0].item()
    return all_categories[c_i], c_i

# print(categoryFromOutput(output))


def randomChoice(l):
    return l[random.randint(0, len(l)-1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)

    return category, line, category_tensor, line_tensor


for i in range(2):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
    # print(category_tensor.shape, line_tensor.shape)


criterion = nn.NLLLoss()

lr = 0.005

def train(category_tensor, line_tensor):
    '''
    '''

    hidden = rnn.initHidden().to(device=device)
    
    rnn.zero_grad()

    for i in range(line_tensor.shape[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-lr * p.grad.data)

    return output, loss.item()


n_iters = 100000
print_every = 5000
plot_every = 1000


current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()


rnn = rnn.to(device=device)


for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    category_tensor = category_tensor.to(device=device)
    line_tensor = line_tensor.to(device=device)

    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


# ----- 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()
