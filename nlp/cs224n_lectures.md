
# cs224n.MIT

- [Lecture_01](#lecture_01)

---

## <div id="lecture_01"></div>Lecture_01


*word vectors* are sometims called *word embeddings* or *word representations*. they are a *distibited representation*.

- one-hot  
    - orthogonal  
    - high dimension  
- word2vec  
    - large corpus of text; fixed vacabulary; get center and context words by sliding window method; use similarity of word vectors to calculate the probability, softmax; update word vector to maximize this probability or minimize negtive log probability
    - every word has two vectors as parameters, one represente center word, another is context word; gradient for center word and context word should needed in descent; easier optimization, average both at end
    - skip-gram  
        - predict context words (position independent) given center word
    - continous bag of word (cbow)  
        - predict center word from bag of context words
    - measurement  
        - syntactic and semantic similarity 
        - computational cost

- Improvement
    - negtive sampling  
        - select negative samples using a 'unigram distribution', where more frequent words are more likely to be selected
        - p(w_i) = f(w_i)^(3/4) / sum_j(f(w_j)^(3/4))
    - sub-sampling of frequent word (eg. the)
        - dont appear as context for reminning word
        - as center word fewer
        - p(w) = (sqrt(z(w)/0.001) + 1) * 0.001 / z(w) to keep

    - word pairs and phrases
        - word2phrase
- one-hot + hidden layer weights (lookup table) --> word vecter
- chain rule (derive gradient)
- stochastic gradient descent

---
reference:
- http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/