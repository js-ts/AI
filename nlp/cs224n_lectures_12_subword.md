# subwords

- phynology
    - human laguage sound
    - phynetic/phynemes

- morphology
    - parts of word with indivisible meaning
    - character n-gram
        - dssm (word hashing)
        - fasttext

- writting system
    - some language no segmentation
    - word-level
        - need to handle
            - large open vovabulary
            - informal spelling (web word)
    - character-level model
        - word embedding can be composed for char embedding
            - bi-lstm
        - handle unknown-word/oov problem
        - slow fully character-level 
    - subword
        - architecture
            - same as word-level model with word-piece
            - hybrid 
        - fasttext
            - ngram augmentation
            - hash-trick
        - wordpiece

- revisiting character-based NMT with capacity and compression (Google AI)
- Bytes pair embedding (BPE)
- wordpiece/sentencepiece-model
    - bert
    - variant of wordpiece model
    - common words are in vocab
        - the, a
    - other words are build from wordpiece
        - hypatia = h ##yp ##ati ##a

- SentencePiece: A simple and language independent subword tokenizer
and detokenizer for Neural Text Processing

- https://github.com/google/sentencepiece
- https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46

