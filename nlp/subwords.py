# byte pair encoding (BPE)
# wordpiece
# unigram-lm
# sentencepiece


# bpe
import os 
import re
import collections
class BPE(object):
    def __init__(self, vocab_size=10):
        self.vocab = {'l o w': 5, 'l o w e r': 2, 'n e w e s t': 6, 'w i d e s t': 1}
        self.num_merges = 10
        self.vocab_size = 10

    def bpe(self, vocab, iters=10):
        for i in range(iters):
            pairs = self.get_state(vocab)
            maxp = max(pairs, key=pairs.get)
            if pairs[maxp] == 1:
                break
            vocab = self.merge_vocab(maxp, vocab)
            print(i, maxp)
        print(vocab)

        return vocab

    @staticmethod
    def get_state(vocab):
        pairs = collections.defaultdict(int)
        for w, freq in vocab.items():
            symbols = w.split()
            for x, y in zip(symbols, symbols[1:]):
                pairs[(x, y)] += freq
        return pairs
    
    @staticmethod
    def merge_vocab(pair, vocab):
        ans = {}
        for w in vocab:
            _w = w.replace(' '.join(pair), ''.join(pair))
            ans[_w] = vocab[w]
        return ans

bpe = BPE()
bpe.bpe(bpe.vocab)



# https://github.com/google/sentencepiece/blob/master/python/README.md
#
def sentencepiece_test():
    import sentencepiece as spm 
    spm.SentencePieceTrainer.Train('--input=./data/docs.txt --model_prefix=xx --vocab_size=100')
    sp = spm.SentencePieceProcessor()
    sp.Load('xx.model')
    pieces = sp.EncodeAsPieces('this is a test')
    ids = sp.EncodeAsIds('this is a test')
    print(pieces)
    print(ids)
