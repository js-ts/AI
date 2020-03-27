# https://www.activestate.com/blog/natural-language-processing-nltk-vs-spacy/

# -------spacy---------------
# spaCy takes an object-oriented approach. Each function returns objects instead of strings or arrays. 
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")
strs = "python is a language in China, Some\nspaces  and\ttab characters"

# Tokenization
# Any sequence of whitespace characters beyond a single space (' ') is included as a token

doc = nlp(strs)
tokens_text = [t.text for t in doc]
print('spacy: ', tokens_text)

tokenizer = Tokenizer(nlp.vocab)
tokens = tokenizer(strs)
print(tokens)
print('spacy: ', [str(t) for t in tokens])


# POS Tagging
print(doc[0].pos, doc[0].pos_)



# Entity Detection
entities = [(i, i.label_, i.label) for i in doc.ents]
print(entities)



# python re: https://www.cnblogs.com/shenjianping/p/11647473.html

