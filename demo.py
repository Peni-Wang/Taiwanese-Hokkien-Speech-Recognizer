# -*- coding: utf-8 -*-
"""## Inference

Need to prepare processor & model & audio & language model(*.arpa)
->
Output prediction

"""

"""### Inference without LM"""
# load processor & moadl
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from wav2vec2_kenlm.decoder import *
import wav2vec2_kenlm.utils
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-large-xlsr-tat/processor") # Input: Processor
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-xlsr-tat/model/checkpoint-37100") # Input: Model
model.to(device)

# predicte
def pred(audio):
	speech, sr = librosa.load(audio, sr = 16000, mono = True)
	input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values.to(device)
	with torch.no_grad():
	  logits = model(input_values).logits
	pred_ids = torch.argmax(logits, dim=-1)
	pred_str = processor.batch_decode(pred_ids)[0]
	return pred_str

file = "./test/sunday.wav" # Input: Test audio file
text = pred(file)
print(text) # Output: Prediction

"""### Inference with LM"""
# prepare vocabulary
lm_path = "./3-gram-tat.arpa" # Input: Language model
vocab_dict = processor.tokenizer.get_vocab()
sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
vocab = []
for _, token in sort_vocab:
    vocab.append(token)
vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '

# prepare LM
def init_kenlm(alpha=0.5, beta=0.5, beam_width=1024):
    beam_decoder = BeamCTCDecoder(vocab[:-2], lm_path=lm_path,
                alpha=alpha, beta=beta,
                cutoff_top_n=40, cutoff_prob=1.0,
                beam_width=beam_width, num_processes=16,
                blank_index=vocab.index(processor.tokenizer.pad_token))
    return beam_decoder

# predicte
def decode_kenlm(audio):
    speech, sr = librosa.load(audio, sr = 16000, mono = True)
    input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    beam, beam_decoded_offsets = beam_decoder.decode(logits)
    return beam

beam_decoder = init_kenlm(alpha=1.5, beta=0.9, beam_width=32) # parameter
decode_output = decode_kenlm(file)
print(decode_output[0][0]) # Output: Prediction

"""### Test 2"""
file = "./test/howmuch.wav"
text = pred(file)
print(text)
decode_output = decode_kenlm(file)
print(decode_output[0][0])