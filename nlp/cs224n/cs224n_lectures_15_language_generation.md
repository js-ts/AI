
# NLG
- refers to any settting in which we generate new text.

- NLG is subcomponet of 
    - Machine translation
    - Abstractive summarization
    - Dialogue (chit-chat and task-based)
    - Creaive writing, storytelling, poetry-generation
    - Freeform QA, answer is generated, not extracted from text or knownledge base
    - Image captioning

- Language Model
    - the task of predicting the next word, given the words so far. $p(y_t/y_1,...,y_{t-1})$
    - A system that produces this probability distribution is called a LM
    - If the system is RNN, its called a RNN-LM

- Conditional LM
    - the task of predicting the next word, given the words so far, and some other input $x$, $P(y_t/y_1,...y_{t-1},x)$
    - Machine translation (x=source sentence, y=target sentence)
    - summarization (x=input text, y=summarized text)
    - Dialogue (x=dialogue history, y=next utterance)

- Training a conditional LM
    - during traing, we feed the gold (aka reference) target sentence into the decoder, regardless of what the decoder predicts. This training method is called **Teacher Forcing**

- Decoder Algorithm
    - is an algorithm you use to generate text from your language model
    - greedy decoding
        - on each step, take the most probable word (argmax)
        - use that as the next word, and feed it as input on the next step
        - keep going until produce #end# or reach some max length

    - beam search
        - which aims to finds a high-probability sequence (not necessarily the optimal sequence, though) by tracking multiple possible sequence at once.
        - on each step of decoder, keep track og the K most probable partial sequences (which we call hypotheses)
            - k is the beam size
        - After you reach some stopping criterion, choose the sequence with the highest probability (factoring in some adjustment for length)
            - $score(y_1,...y_t)=\sum_{i=1}^t logP_{LM}(y_i/y_1,...,y_i-1,x)$

        - effect of beam size in chitchat dialogue
            - low beam size: more on-topic but nonsensical; bad english
            - high beam size: converges to safe, 'correct' response, but it's generic and less relevant.

    - sampling-based decoding
        - pure sampling
            - on each step t, randomly sample from the probability distribution P_t to obtain your next word

        - top-n sampling
            - on each step t, randomly sample fron P_t restricted to just the top-n most probable words
            - n=1 is greedy search, n=V is pure sampling
            - increase n to get more diverse/risky output
            - decrease n to get more generic/safe output
    
    - softmax temperature
        - scores $s\in R^{V}$
        - $P_\iota{(w)}=\frac{exp(s_w/\iota)}{\sum_{w'\in V}exp(s_{w'}/\iota)}$
        - raise the temperature
            - P becomes more uniform
            - thus more diverse output, probability is spread around vocab
        - lower the temperature
            - P becomes more spiky
            - thus less diverse, probability is concentrated on top words


## NLG tasks and neural approaches to them

- Summarization
    - given input text x, write a summary y which is shorter and contains the main information of x
    - summarization can be single/multi-document
    - extractive ~
        - select parts of the original text to form a summary
    - abstractive ~
        - generate new text using nlg techniques
    - ROUGE
        - like BLUE, it's based on n-gram overlap
        - x

---
- Metrics