
# cs224n.MIT

|lecture | outline|
|-|-|
[Lecture_01](#lecture01) | word2vec, sg
[Lecture_02](#lecture02) | glove, evaluate


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

    - Hierarchy Softmax  
    - sub-sampling of frequent word (eg. the)
        - dont appear as context for reminning word
        - as center word fewer
        - p(w) = (sqrt(z(w)/0.001) + 1) * 0.001 / z(w) to keep

    - word pairs and phrases
        - word2phrase

- chain rule (derive gradient)
- stochastic gradient descent

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
reference:
- http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
- https://blog.csdn.net/sinat_29819401/article/details/90669304
