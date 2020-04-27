
# Contextual word embedding

- recap
    - unsupervised pretrain
    - fineturn
    - word embedding
        - word2vec
        - glove
        - fasttext

- unknown word
    - \<unk\> for rarer words
        - collapse
    - char-level model
    - sign random vector

- fixed representation problem
    - without context
    - word have different aspects, semantic and syntactic

- ELMo
    - word representation using char-cnn
    - two layer biLSTM with residual
    - train lm
    - elmo embedding <= word embedding + hidden1 + hidden2
    - lower-layer syntax and higher-level semantics

- ULMfit
- 