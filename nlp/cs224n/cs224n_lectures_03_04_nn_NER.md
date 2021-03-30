
## Named Entity Recognition


- NER
    - find and classify names in text
    - ORG / LOC / PER / MISC

- NER on word sequences
    - predict entities by classifying words in context and then extracting entities as word subseqences
    - hard to work out boundaries of entity
    - hard to know if something is an entity
    - hard to know class of unknown/novel entity
    - entity class is ambiguous and depends on context

- word window classification
    - classify a word in its contexy window of neighboring words
    - *average* (lose position information)
    - *concatenation* of word vectors surrounding it in a window
    