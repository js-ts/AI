
# cs224n.MIT

|lecture | outline|
|-|-|
[Lecture_01](#lecture01) | word2vec, sg
[Lecture_02](#lecture02) | glove, evaluate
[fastText](#lecture02) | fastText, 

---

## <div id="lecture01"></div>Lecture_01

*word vectors* are sometims called *word embeddings* or *word representations*. they are a *distibited representation*.

### discrete representation
- one-hot  
    - orthogonal  
    - high dimension (depends on vocabulary) 
    - hash trick (naive) 


### distributional representation
- word2vec
    - skip-gram  
        - predict context words (position independent) given center word
        - large corpus of text; fixed vacabulary; get center and context words by sliding window method; use similarity of word vectors to calculate the probability, softmax; update word vector to maximize this probability or minimize negtive log probability
        - every word has two vectors as parameters, one represente center word, another is context word; gradient for center word and context word should needed in descent; easier optimization, average both at end
        - one-hot + hidden layer weights (lookup table) -> word vecter -> softmax (prob)

    - continous bag of word (cbow)  
        - predict center word from bag of context words
    - measurement  
        - syntactic and semantic similarity 
        - computational cost

- Improvement
    - softmax -> sigmoid (logistic regression)
    - negtive sampling  
        - select negative samples using a 'unigram distribution', where more frequent words are more likely to be selected. 
        - p(w_i) = f(w_i)^(3/4) / sum_j(f(w_j)^(3/4)), the power makes less frequent words be sampled more often  
    - Hierarchical Softmax  
    - sub-sampling of frequent word (eg. the)
        - dont appear as context for reminning word
        - as center word fewer
        - p(w) = (sqrt(z(w)/0.001) + 1) * 0.001 / z(w) to keep

    - word pairs and phrases
        - word2phrase

- chain rule (derive gradient)
- stochastic gradient descent


reference:
5. https://towardsdatascience.com/hierarchical-softmax-and-negative-sampling-short-notes-worth-telling-2672010dbe08

---

## <div id="lecture02"></div>Lecture_02


### Count-based Method

- Co-ocurrence matrix
    - window vs. full document
    - window, similar to word2vec, SK and CBOW
    - word-document co-occurence matrix will given general topics leading to 'Latent Semantic Analysis'
- GloVe
    - Matrix: x_ij represents count of x_i as center word and x_j is context word during corpus
    - p_ij is p(j | i) = x_ij / x_i 
    - J = sum_ij(f(x_ij) * (w_i.T dot w_j + b_i + b_j - log(x_ij)) ^ 2)


- Evaluate word vector
    - intrinsic
        - semantic
        - syntactic
    - extrinsic


---

### fastText

- one majar drawback for above methods were its inability to deal with out of corpus words. Because they treat word as the minimal entrity
- character n-grams
    - treat each word and composed of n-grams
    - #india#: #in ind ndi dia ia#
- memory requirement
    - control minimum letter n-grams
    - minimum word count
    - hash n-grams

- Hierarchical softmax
    - vocab too large
    - HuffmanTree
- Multi-label classification
    - softmax 
        - number of labels to predict 
        - the threshold for the predicted probability
    - A convenient way to handle multiple labels is to use independent binary classifiers for each label


reference:
1. https://fasttext.cc/docs/en/supervised-tutorial.html
2. https://ruder.io/word-embeddings-1/index.html
3. http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
4. https://blog.csdn.net/sinat_29819401/article/details/90669304
