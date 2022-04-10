# **Taiwanese-Hokkien-Speech-Recognizer**
Speech recognition system for Taiwanese Hokkien using wav2vec2-xlsr model

# Introduction
* preprocess.py  
    Data preprocessing  

* train.py  
    Fine-tuning wav2vec2 model for Taiwanese Hokkien  

* test.py  
    Computing error rate without/with 3-gram LM  

* demo.py  
    Inference without/with 3-gram LM  

# Performance
Experiments using 41 hours Taiwanese-Hokkien corpus achieve 13% CER. When adding 3-gram language model achieve 7% CER.

# Reference
* Paper:  
    1. [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)  
    2. [Unsupervised Cross-lingual Representation Learning for Speech Recognition](https://arxiv.org/abs/2006.13979)

* Source code:  
    1. [Huggingface Transformers](https://github.com/huggingface/transformers)  
    2. [Fine-tune wav2vec2-xlsr](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)  
    3. [KenLM](https://github.com/kpu/kenlm)  
    4. [wav2vec2-kenlm](https://github.com/farisalasmary/wav2vec2-kenlm)