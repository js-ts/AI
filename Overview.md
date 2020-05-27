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

---
## <div id="cv"></div>Computer Vision
- Classification / Fine-graind Recognition  
- Object / Instance / Keypoint Detection    
- Semantic / Instance Segmentation   
- Multi- and Single- Object Tracking  
- Image Retrival / Search   
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


## <div id="recm"></div>Recommender Systems
- Recommender
- Ads

- DSSM
    - DSSM
    - CDSSM
- [Google YouTube Recommendations]()

Name | commits | resource
--- | --- | ---
DSSM | cdssm | 
[Google YouTube Recommendations]() | candidate generation and ranking | --
[Facebook Practical Lessons from Predicting Clicks on Ads](http://papers.nips.cc/paper/2666-an-investigation-of-practical-approximate-nearest-neighbor-algorithms.pdf) | -- | --
[An Investigation of Practical Approximate Nearest Neighbor Algorithms](http://papers.nips.cc/paper/2666-an-investigation-of-practical-approximate-nearest-neighbor-algorithms.pdf) | k-NN, LSH

## <div id="acc"></div>Acceleration
- Quantization  
- Pruning  
- Distilling  

|ID|Name|Conference|Commence  
|---|---|---|---|
-- | -- | -- | 
