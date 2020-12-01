# AI Overview

- [Tutorials](#tutorial)
- [Machine Learning](#ml)
- [Deep Neural Network](#dnn)
- [Reinforcement Learning](#rl)
- [Generative Adversarial Network](#gan)
- [Natural Language Processing](#nlp)
- [Computer Vision](#cv)
- [Recommender Systems](#recm)
- [Acceleration](#acc)


## <div id="tutorial"></div>Resource, Tutorial, and Course
- Python
    - [My Leecode](https://github.com/lyuwenyu/Leetcode)
- PyTorch
    - [HomePage](https://pytorch.org/)

- Mathmatics
    - Calculus  
        - [matrixcalculus](http://www.matrixcalculus.org/)
    - Statistics  
    - Probability  
- Etc. 

ID|name|Commence  
---|---|---
00 | [cs231n.stanford](http://cs231n.stanford.edu/) | CS231n: Convolutional Neural Networks for Visual Recognition
01 | [cs224n.stanford](http://web.stanford.edu/class/cs224n/) | CS224n: Natural Language Processing with Deep Learning
02 | [RL Course by David Silver](https://www.youtube.com/playlist?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-) | RL Course by David Silver
03 | [DLRLSS 2019](https://www.youtube.com/watch?v=O2o4oONWCWA&list=PLKlhhkvvU8-aXmPQZNYG_e-2nTd0tJE8v&index=2&t=0s) | Amii Intelligence, DL and RL Summer School 2019
04 | [MIT 18.065](https://ocw.mit.edu/courses/mathematics/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/syllabus/18.065-course-introduction/) | Linear Algebra Gilbert Strang  
05 | [UFLDL]() | by Andrew Ng


## <div id="ml"></div>Machine Learning   
- ML  

|ID|Name|Conference|Commence  
|---|---|---|---|
-- | -- | -- | --


## <div id="dnn"></div>Deep Neural Network
- Architecture
    - FullyConnect
    - Convolution
        - group
        - dilate
        - separable
            - depth/point-wise
    - Pooling
        - ave/max/kmax/
    - Activation
        - tanh/sigmoid/relu/prelu/swish/
    - Normalize
        - bn/gn/ln/in/
        - l1/l2/
        - evolving normalization-activation
    - Dropout
    - [torch.nn]()

- Optimization
    - BP
        - chain-rule
    - Gradient
        - vanishing/memory/skipconnect
        - exploding/clipping
    - SGD/Adam/AdamW
    - Loss-Criterion
    - LR-Schedule
    - [torch.optim]()

- CNN  
    - LeNet
    - AlexNet
    - VGGNet
    - GoogleNet/Inception/MobileNet
    - ResNet/DenseNet/ReNeXt
    - EfficientNet
    - RegNet
    - [torchvision]()

- RNN
    - Vanilla
    - Gated
        - LSTM/GRU/biLSTM
    - Q-RNN 
    - BPTT

- NN from scrach  
    - [tiny example](https://github.com/lyuwenyu/AI/tree/master/nn)
        - forward
            - specific funcion
            - record tensors for bp
        - backward
            - delta error 
            - derive according specific function to inputs and weights
        - update
            - grident
            - laerning rate
            - momentum
            - weight decay

- Summary
    - Defination 
        - metrics
    - Solving
        - Data
        - Label
        - Feature
        - Model
        - Optimization
    - Serving
        - xxx

|ID|Name|Conference|Commence  
|---|---|---|---|
-- | [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) | -- | gradient vanishing and exploding
-- | [Bag of Tricks for Image Classification with Convolutional Neural Networks]() | -- | --
-- | [Neural Network Optimization](https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0) | blog | --
-- | [CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization](https://arxiv.org/abs/2004.15004) | -- | --


## <div id="gan"></div>Generative Adversarial Network
- GAN  
    - Generator
    - Discriminator

|ID|Name|Conference|Commence  
|---|---|---|---|
-- | -- | -- | --


## <div id="rl"></div>Reinforcement Learning
- RL
- [My practice](https://github.com/lyuwenyu/RL)

ID|Name|Conference|Commence  
---|---|---|---
00 | [Reinforcement Learning: An Introduction (Second Edition)](http://www.incompleteideas.net/book/RLbook2018.pdf) | -- | --
01 | [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) | -- | --
02 | [Stanford CS234: Reinforcement Learning](http://web.stanford.edu/class/cs234/index.html) | -- | --
03 | [CS 294-112 at UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/) | -- | --

## <div id="nlp"></div>Natural Language Processing

- Preprocessing 
    - Normalization/Pretokenizer
        - text-encoding/lowercase/unicode/byte
        - cleaning/canonical
        - split/byte-level/whitespace/chardelimiter
    - Segmentation/Tokenization/Model
        - morphology/
        - word/character/byte
        - subword/bpe/unigram-lm/wordpiece
    - Numbericalization/Postprocessor
        - map/hash-id
        - problem/oov/overfitting
        - bertprocess
    - Decode
        - word/bpe/wordpiece
    - Tools
        - [spacy]()
        - [sentencepiece](https://github.com/google/sentencepiece)
        - [tokenizers](https://github.com/huggingface/tokenizers)

- Embedding
    - word/character/sentence/document/
    - distributional representation
        - counting/predicting
        - word2vec
            - cbow
            - skip-gram
        - glove
        - fasttext
    - contexture representation
        - pretrained-nn/transfer/fineturn
        - ELMo
        - transformers

- Language Model
    - statistical methods
    - generation-p
    - recurrent-nn
    - ELMo/ULMfit
    - Attention/Transformer/Transformer-XL
    - BERT/RoBERTa/m-lm/ae-lm
    - GPT/GPT-2
    - XLNet/p-lm/ar-lm
    
- Machine Translation
    - transduction
    - seq2seq
    - attention/transformer
    - Google-NMT
    
- Natual Language Generation
    - NMT
    - Summarization
    - Dialogue
    - Image Captioning
    - Freeform QA
    - Decoding
        - Greedy/Beam-seach
        - Pure/TopK-sampling
    - [fairseq](https://github.com/pytorch/fairseq)

- Classification
    - word/text/sentiment/
    - bow/rnn/window/cnn/

- Question Answer
    - reading comprehension
    - extracted from tex

- Others 
    - Linguistic structure
        - dependency/constituency-parsing
    - tokenization
    - part-of-speech tagging
    - named entity recognition
 
Paper | Commits | Related  
---|---|---
[Processing]() | -- | [the wonderful world of preprocessing in nlp](https://mlexplained.com/2019/11/06/a-deep-dive-into-the-wonderful-world-of-preprocessing-in-nlp/)
[Word2vec]()| distributionl representation; cbow, skip-gram | [How to Train word2vec](http://jalammar.github.io/illustrated-word2vec/)
[GloVe]() | global information, coocurrent maxtrix|
[fastText]() | oov problem, letter n-gram | 
[Character-Aware](https://arxiv.org/pdf/1508.06615.pdf) | character-based, char-embedding->cnn->pool | 
[ELMo]() | BiLM; contextual, deep, character-based; [embedding, hidden1, hidden2] | 
[Transformer](https://arxiv.org/pdf/1706.03762.pdf) | [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html#batches-and-masking) | 
[GPT]() | | [gpt-2]()
[BERT](https://arxiv.org/pdf/1810.04805.pdf) | autoencoder(AE) language model; masked-lm; [bert-research by mc](http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/#31-input-representation--wordpiece-embeddings); pre/post-norm | [RoBERTa: A Robustly Optimized BERT](https://arxiv.org/pdf/1907.11692.pdf); [Extreme language model compression with optimal subwords and shared projections](https://arxiv.org/pdf/1909.11687.pdf)
[XLNet]() | autoregressive(AR) language model; Permutation Language Modeling; Two-Stream Self-Attention; [What is XLNet and why it outperforms BERT](https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335); [xlnet-theory](http://fancyerii.github.io/2019/06/30/xlnet-theory/)
[Efficient Transformers: A Survey](https://arxiv.org/pdf/2009.06732.pdf)| -- | -- 


---
## <div id="cv"></div>Computer Vision
- Classification
    - Recognition
    - Fine-graind Recognition  
- Detection
    - Object 
    - Instance
    - Keypoint     
- Segmentation
    - Semantic
    - Instance    
- Tracking
    - Single Object
    - Multiple Object
- Image Search   
- Style Transform
- Optical Flow  
- 3D-restruction 
    - Calibration 
    - Motion Pose 
    - Depth
    - SLAM
- Mine
    - [pytorch-workspace](https://github.com/lyuwenyu/pytorch_workspace)
    - [Geometry]()
    - [Detetron]()

Name | commits | resource
--- | --- | ---
[DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf) | -- | --
[End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf) | DETR | --

## <div id="recm"></div>Recommender Systems
- Recommender
- Ads

- GOOGLE
    - [Google YouTube Recommendations]() 
    - Deep & Wise & Cross
    
- MICROSOFT
    - DSSM
    - CDSSM
    
- FACEBOOK
    - DLRM
    
- ALIBABA
    - DIN 
    - DIEN

Name | commits | resource
--- | --- | ---
[Google YouTube Recommendations]() | Marching, Ranking | --
[Learning Deep Structured Semantic Models for Web Search. Microsoft]() | CDSSM | --
[Practical Lessons from Predicting Clicks on Ads at Facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf) | GBDT + LR | --
[Wide & Deep Learning for Recommender Systems. Google](https://arxiv.org/pdf/1606.07792.pdf) | Wide + Deep | -- 
[Deep & Cross Network for Ad Click Predictions. Google](https://arxiv.org/pdf/1708.05123.pdf) | DCN | --
[Deep Learning Recommendation Model for Personalization and Recommendation System. Facebook](https://arxiv.org/pdf/1906.00091.pdf) | DLRM | --
[Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf) | DIN DIEN | --
[Category-Specific CNN for Visual-aware CTR Prediction at JD.COM](https://arxiv.org/pdf/2006.10337.pdf) | -- | --

## <div id="acc"></div>Acceleration
- Quantization  
- Pruning  
- Distilling  

|ID|Name|Conference|Commence  
|---|---|---|---|
-- | -- | -- | 
