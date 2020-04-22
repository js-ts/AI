# byte pair encoding (BPE)
# wordpiece
# unigram-lm
# sentencepiece
# https://huggingface.co/blog/how-to-train
# https://www.kaggle.com/funtowiczmo/hugging-face-tutorials-training-tokenizer


# bpe
import os 
import re
import collections

class BPE(object):
    def __init__(self, vocab_size=10):
        self.text = {'l o w': 5, 'l o w e r': 2, 'n e w e s t': 6, 'w i d e s t': 1}
        self.num_merges = 10
        self.vocab_size = 10
        self.vocab = {}

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

# bpe = BPE()
# bpe.bpe(bpe.text)


# sentencepiece
# https://github.com/google/sentencepiece/blob/master/python/README.md
#
def sentencepiece_test():
    import sentencepiece as spm 
    spm.SentencePieceTrainer.Train('--model_type=bpe --input=train.en --model_prefix=xx --vocab_size=2000')
    # spm.SentencePieceTrainer.Train('--model_type=unigram --input=./data/docs.txt --model_prefix=xx --vocab_size=50')
    sp = spm.SentencePieceProcessor()
    sp.Load('xx.model')

    # vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    # print([x for x in vocab if len(x) == 1])
    # print(vocab)

    pieces = sp.encode_as_pieces('Two young, White males are outside near many bushes.')
    # ids = sp.encode_as_ids('like')
    # words = sp.decode_ids([59])
    print(pieces)

# sentencepiece_test()


# tokenizers
def tokenizers_test():
    from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer
    from tokenizers.processors import BertProcessing
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import Lowercase, Sequence
    
    # tokenizer = BertWordPieceTokenizer('./data/vocab.txt', lowercase=True)
    # output = tokenizer.encode('hello, likee')
    # print(output.ids, output.tokens, output.offsets)

    tokenizer = BertWordPieceTokenizer()
    # tokenizer = BertWordPieceTokenizer(lowercase=True)
    # tokenizer = CharBPETokenizer(lowercase=True)
    tokenizer.train(['./data/train.en'], vocab_size=5000, )
    tokenizer.save('./data', 'xx')

    # tokenizer = ByteLevelBPETokenizer('./data/xx-vocab.json', './data/xx-merges.txt')
    # tokenizer._tokenizer.post_processor(BertProcessing(("</s>", tokenizer.token_to_id("</s>")),("<s>", tokenizer.token_to_id("<s>")),))
    tokenizer = BertWordPieceTokenizer('./data/xx-vocab.txt')
    print(tokenizer.encode('Two young, White males are outside near many bushes.').tokens)

tokenizers_test()


