import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SkipGramModle(nn.Module):
    '''
    u : center word
    v : context word
    - negtive sampling
    '''
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModle, self).__init__()

        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        init.uniform_(self.u_embeddings.weight.data, -1.0/emb_dimension, 1.0/emb_dimension)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        '''
        '''
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_v_neg = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, min=-10, max=10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_v_neg, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return (score + neg_score).mean()


# optimizer = optim.SparseAdam(skip_gram_model.parameters(), lr=self.initial_lr)


# glove
