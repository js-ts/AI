
# machine translation


machine translation is a task of translating s sentence x from one language to a sentence y in another language.

- rule-based
- statistical mt
    - p(y|x)
    - argmax_y(p(x|y) p(y))
    - translation model p(x|y)
    - language model p(y)
- neural mt
    - p(y|x)
    - encoder produce an embedding ogf the source sentence
    - decoder is a language model the generating target sentence conditioned on encoding
    - greedy decoding
    - beam search decoding
        - keep track of k most probable partial translations
        - score = log p(y1...yt | x) = sum_i log_p(yi | y1...yi-1, x)
        - not guaranteed to find optimal solution
        - problem longer hypotheses have lower scores
        - normalize by lenght, scores / t
    
    - advantages
        - end-to-end
        - better use of context
        - better use of phrase similarities
    
    - disadvantage
        - less interpretable
        - hard to debug
        - difficult to control

- Evaluate
    - BLUE bilingual evaluation understudy
        - compare machine-written to one or more human-written and compute a similarity score based on
            - n-gram precision
            - plus a penalty for too short system translations
    - 


- seq2seq
    - conditional language model

- attention
    - core idea on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence
    - 


