# -------nltk-------
# NLTK is essentially a string processing library, where each function takes strings as input and returns a processed string. 

import nltk

strs = "python is a language in China, Some\nspaces  and\ttab characters"
tokens = nltk.word_tokenize(strs)
print('Tokenization: ', tokens)


postagging = nltk.pos_tag(tokens)
print("POS Tagging: ", postagging)


# Entity Detection
print(nltk.ne_chunk(postagging, binary=True))
