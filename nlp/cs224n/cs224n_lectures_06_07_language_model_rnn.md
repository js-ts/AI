
# Language Modeling

Language Modeling is the task of predicting what word comes next.

- n-gram language model
    - a n-gram is a chunk of n consecutive words.  
        - [ the students open their _ ]
        - unigrams: 'the', 'students', 'open'
        - bigrams: 'the students', 'students open' 
    - idea: collect statistics about how frequent different n-grams are, and using these to predict next word.
    - problems
        - sparse
        - storage  

- Fixed window neural network

- Recurrent Neural Network
    - core idea: apply the same weights w repeatedly
    - can process any length input  
    - can (in theory) use information from many steps back
    - model size is fixed
    - same weights applied on every timestep, so there is symmetry in how inputs are precessed
    - slow, can not parallel
    - difficult to access imformation from many steps back

- Gradient
    - vanishing
        - gradient signal from faraway is lost because its much smaller than gradient signal from close-by
        - gradient can be viewed as a meature of the effict of the past on the future
        - rnn-lm are better at learning from sequential recency than syntactic recency
        - memory

    - exploding
        - if the gradient becomes too big, then the SGD update step becomes too big
        - Inf / NaN
        - Solution: gradient clipping
        - Intuition: take a step in th same direction, but a smaller step

- Evaluating LM
    - perplexity


---

- LSTM: long short-term memory
    - hidden state: output_tanh(c(t))
    - cell state: forget_c(t-1) + input_(c_new)
    - new cell content: h(t-1), x(t) 
    - forget gate: h(t-1), x(t) 
    - input gate: h(t-1), x(t)
    - output gate: h(t-1), x(t)

- GRU: gated recurrent units
    - hidden state: (1-update)(h(t-1)) + update(h_new)
    - new hidden state content: reset_h(t-1), x(t) 
    - update gate: h(t-1), x(t) 
    - reset gate: h(t-1), x(t) 

- Bidiractional RNNs
    - foreward
    - backward
    - concatenate
- Multi-layer RNNs
    - the hidden states from layer i are inputs to layer i+1