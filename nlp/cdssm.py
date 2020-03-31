
import torch
import torch.nn as nn

class CDSSM(nn.Module):
    def __init__(self, TEXT, letter_vocab,):
        super(CDSSM, self).__init__()

        self.word_n_gram = 3
        self.triletter_dims = len(letter_vocab)
        self.n_words = len(TEXT.vocab)
        self.word_dim = self.word_n_gram * self.triletter_dims

        self.K = 300
        self.L = 128

        self.embedding = nn.Embedding(self.n_words, self.triletter_dims)
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.requires_grad = False    

        self.left_conv1d = nn.Conv1d(self.word_dim, self.K, kernel_size=3, padding=1)
        self.left_semantic = nn.Linear(self.K, self.L)

        self.right_conv1d = nn.Conv1d(self.word_dim, self.K, kernel_size=3, padding=1)
        self.right_semantic = nn.Linear(self.K, self.L)

        # self.maxpool1d = nn.MaxPool1d()
        self.cossim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, src, trg, negs=None):
        '''
        [len, batch, dim]
        '''
        # src = self.embedding(src)
        # _, _, dim = src.shape
        # src = [torch.cat(t, dim=-1) for t in zip(*[src[i:] for i in range(self.word_n_gram)])]
        # src = torch.cat(src, dim=0).view(-1, bss, self.word_n_gram * dim)
        src = self._embedding(src)
        src = src.permute(1, 2, 0) # batch dims lens
        src = torch.tanh(self.left_conv1d(src))
        src = torch.max(src, dim=-1)[0]
        src = torch.tanh(self.left_semantic(src))
        # print('src semantic: ', src.shape)

        trg = self._embedding(trg)
        trg = trg.permute(1, 2, 0) # -> batch dims lens
        trg = torch.tanh(self.right_conv1d(trg))
        trg = torch.max(trg, dim=-1)[0]
        trg = torch.tanh(self.right_semantic(trg))
        # print('trg semantic: ', trg.shape)

        negs = [self._embedding(neg) for neg in negs]
        negs = [neg.permute(1, 2, 0) for neg in negs]
        negs = [torch.tanh(self.right_conv1d(neg)) for neg in negs]
        negs = [torch.max(neg, dim=-1)[0] for neg in negs]
        negs = [torch.tanh(self.right_semantic(neg)) for neg in negs]

        sims = [self.cossim(src, trg)]
        sims += [self.cossim(src, neg) for neg in negs]

        sims = torch.cat([sim.view(-1, 1) for sim in sims], dim=-1)
        # print(sims.shape)

        return sims

    def _embedding(self, data):
        data = self.embedding(data)
        lens, bss, dim = data.shape
        data = [torch.cat(t, dim=-1) for t in zip(*[data[i:] for i in range(self.word_n_gram)])]
        data = torch.cat(data, dim=0).view(-1, bss, self.word_n_gram * dim)
        return data




"""

# Iterator
bs = 8
n_negs = 4
epoches = 10

print('TEXT.vocab.vectors.shape ', TEXT.vocab.vectors.shape)
embedding = nn.Embedding(len(TEXT.vocab), len(letter_vocab))
embedding.weight.data.copy_(TEXT.vocab.vectors)
embedding.weight.requires_grad = False

device = torch.device('cuda')
cdssm = CDSSM().train().to(device=device)

train_iter = BucketIterator(dataset=train, batch_size=bs, sort_key=lambda x: len(x.src), device=device, shuffle=True)
neg_train_iter = BucketIterator(dataset=train, batch_size=1, sort_key=lambda x: len(x.src), device=device, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([p for p in cdssm.parameters() if p.requires_grad], lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

embedding = embedding.to(device=device)

for e in range(epoches):

    for i, batch in enumerate(train_iter):
        src = batch.src # lens, batch, dims
        trg = batch.trg

        src = embedding(src).to(device=device)
        trg = embedding(trg).to(device=device)

        # negtive 
        negs = []
        for j in range(n_negs):
            ns = []
            for bi in range(trg.shape[1]):
                _x = src[:, bi:bi+1, :]
                _y = trg[:, bi:bi+1, :]
                for _bs in neg_train_iter:
                    _src = embedding(_bs.src)
                    _trg = embedding(_bs.trg)
                    if abs(_trg.sum() - _y.sum()) > 3 and abs(_src.sum() - _x.sum()) > 3: # other simlarity metric
                        ns += [_trg]
                        break
            minl = min([x.shape[0] for x in ns])
            negs += [torch.cat([x[:minl] for x in ns], dim=1).to(device=device)]

        sims = cdssm(src, trg, negs)
        sims = F.softmax(sims, dim=-1)
        loss = -torch.log(sims[:, 0]).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(e, i, loss.item())

    scheduler.step()

"""