import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np 
from torch.nn.modules.sparse import EmbeddingBag

import fasttext 
import fasttext.util

__all__ = ['SkipGramModle', 'HierarchicalSoftmax', 'FastText', '_fnvhash']


class SkipGramModle(nn.Module):
    '''
    u : center word
    v : context word
    - negtive sampling (like: GAN, discriminator; SGD; Unigram distribution; p ^ 3/4)
    - hierarchical softmax
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


# HierarchicalSoftmax
# reference: https://github.com/leimao/Two_Layer_Hierarchical_Softmax_PyTorch
# reference: https://github.com/OpenNMT/OpenNMT-py/issues/541

class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, ntokens_per_class = None):
        super(HierarchicalSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nhid = nhid

        if ntokens_per_class is None:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))

        self.ntokens_per_class = ntokens_per_class

        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        self.ntokens_actual = self.nclasses * self.ntokens_per_class

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)

        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)

    def forward(self, inputs, labels = None):
        batch_size, _ = inputs.size()
        if labels is not None:
            label_position_top = labels / self.ntokens_per_class
            label_position_bottom = labels % self.ntokens_per_class

            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + self.layer_bottom_b[label_position_top]
            layer_bottom_probs = self.softmax(layer_bottom_logits)

            target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]

            return target_probs

        else:
            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)
            word_probs = layer_top_probs[:,0] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0])
            for i in range(1, self.nclasses):
                word_probs = torch.cat((word_probs, layer_top_probs[:,i] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[i]) + self.layer_bottom_b[i])), dim=1)

            return word_probs


# fastText
#
class FastTextEmbedding(nn.EmbeddingBag):
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)
        input_matrix = self.model.get_input_matrix()
        super().__init__(*input_matrix.shape)
        self.weight.data.copy_(torch.FloatTensor(input_matrix))

    def forward(self, words):
        word_subinds = np.empty([0], dtype=np.int64)
        word_offsets = [0]
        for word in words:
            _, subinds = self.model.get_subwords(word)
            word_subinds = np.concatenate(word_subinds, subinds)
            word_offsets.append(word_offsets[-1] + len(subinds))
        word_offsets = word_offsets[:-1]
        ind = torch.tensor(word_subinds).to(torch.long)
        offsets = torch.tensor(word_offsets).to(torch.long)
        
        return super().forward(ind, offsets)


class FastText(object):
    '''equal to fasttext.get_word_vector('word')
    '''
    def __init__(self, path='cc.en.300.bin'):
        if path.endswith('bin'):
            ft = fasttext.load_model(path)
            self.weights = ft.get_input_matrix()
            self.words = ft.get_words()
        elif path.endswith('npy'):
            self.weights = None
            self.words = None

        self.nwords = len(self.words)
        self.nsubwords = self.weights.shape[0] - self.nwords
        self.dims = self.weights.shape[-1]
        self.ngrams = 5
        
        assert self.nwords == 2000000, ''
        assert self.nsubwords == 2000000, ''
        assert self.dims == 300, ''
    
    def get_word_vector(self, w):
        ''''''
        _, ids = self._get_subwords(w)
        return self.weights[ids, :].mean(axis=0)        

    def _get_subwords(self, w):
        ''''''
        subws = zip(*[list('<'+w +'>')[i:] for i in range(self.ngrams)])
        subws = [''.join(x) for x in subws]
        subws_ids = [self._get_subword_id(x) for x in subws]
        w_id = self._get_word_id(w)
        if w_id != -1:
            subws = [w] + subws
            subws_ids = [w_id] + subws_ids
        return subws, subws_ids

    def _get_subword_id(self, subw):
        return self.fnvhash(subw) % self.nsubwords + self.nwords

    def _get_word_id(self, w):
        try:
            return self.words.index(w)
        except:
            return -1

    @staticmethod
    def fnvhash(data, init=0x811c9dc5, prime=0x01000193, size=2**32):
        '''fnva
        reference: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
        '''
        val = init
        for b in data:
            val = val ^ ord(b)
            val = (val * prime) % size # uint_32
        return val
    

# reference:
# https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py

"""
uint32_t hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(int8_t(str[i]));
    h = h * 16777619;
  }
  return h;
}
"""

def _fnvhash(data, init=0x811c9dc5, prime=0x01000193, size=2**32):
    val = init
    for b in data:
        val = val ^ ord(b)
        val = (val * prime) % size
    return val

# val = _fnvhash('word')
# print(val)
