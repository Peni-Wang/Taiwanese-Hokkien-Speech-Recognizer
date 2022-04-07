# -*- coding: utf-8 -*-
"""## Test

Need to prepare processor & model & audio & language model(*.arpa)
->
Output error rate

"""

"""### Test without LM"""
# load test data
from datasets import load_dataset
test = load_dataset('json', data_files='./test.json')

# wav to array
import soundfile as sf
import os
datapath = "./TAT"
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = sf.read(os.path.join(datapath, batch["path"]))
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch
test = test.map(speech_file_to_array_fn, remove_columns=test.column_names["train"], num_proc=3)

# load model
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-large-xlsr-tat/processor")
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-xlsr-tat/model/checkpoint-37100")

# predicte
import torch
def map_to_result(batch):
  model.to("cuda")
  input_values = processor(
      batch["speech"], 
      sampling_rate=16000, 
      return_tensors="pt"
  ).input_values.to("cuda")

  with torch.no_grad():
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  
  return batch
results = test["train"].map(map_to_result, num_proc=2)

# compute SER & CER
from datasets import load_metric
wer_metric = load_metric("wer")
print("Test SER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["target_text"])))

cer_metric = load_metric("cer")
print("Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], references=results["target_text"])))

"""### Test with LM"""
# load LM
from wav2vec2_kenlm.decoder import *
import wav2vec2_kenlm.utils
from transformers import Wav2Vec2CTCTokenizer

vocab_dict = processor.tokenizer.get_vocab()
sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
vocab = []
for _, token in sort_vocab:
    vocab.append(token)
vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '

# prepare LM
lm_path = "./3-gram-tat.arpa"
alpha = 0.5
beta = 0.5
beam_width = 1024

greedy_decoder = GreedyDecoder(vocab, blank_index=vocab.index(processor.tokenizer.pad_token))

beam_decoder = BeamCTCDecoder(vocab[:-2], lm_path=lm_path,
            alpha=alpha, beta=beta,
            cutoff_top_n=40, cutoff_prob=1.0,
            beam_width=beam_width, num_processes=16,
            blank_index=vocab.index(processor.tokenizer.pad_token))

# predicte
def decode_kenlm(batch):
    speech, sr = sf.read(os.path.join(datapath, batch["path"]))
    input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values.to("cuda")
    with torch.no_grad():
        logits = model(input_values).logits
    batch["greedy"], greedy_decoded_offsets = greedy_decoder.decode(logits)
    batch["beam"], beam_decoded_offsets = beam_decoder.decode(logits)
    
    return batch
decode_output = test["train"].map(decode_kenlm)

# compute CER
GS = np.array(decode_output["greedy"]).squeeze()
print("Test CER(Greedy): {:.3f}".format(cer_metric.compute(predictions=GS, references=decode_output["target_text"])))

BS = np.array(decode_output["beam"]).squeeze()
BST = BS.T
print("Test CER(Beam): {:.3f}".format(cer_metric.compute(predictions=BST[0], references=decode_output["target_text"])))