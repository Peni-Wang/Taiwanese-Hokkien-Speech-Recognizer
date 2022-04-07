# -*- coding: utf-8 -*-
"""## Preprocessing

Need to prepare json files(audio path, transcribe) & regex.txt(punctuation)
->
Output train.json & test.json {path, text} & vocab.json {character: token id}

"""

"""### Create tat.json"""
import os
datapath = './TAT/' # corpus root directory
jsons = [] # every json files name, including path

# visit and list all files in ./TAT/json directory
alllist = os.walk(os.path.join(datapath, "json"))
for root, dirs, files in alllist:
  if files:
    for json in files:
      jsons.append(os.path.join(root, json))
print("Json path: ", jsons[0])

# read all json file to huggingface dataset
from datasets import load_dataset
tat = load_dataset('json', data_files=jsons) 
print("TAT: ", tat)
print("text0: ", tat['train']['漢羅台文'][0])

# convert to audio id & speaker id & text, and remove other not necessary columns
def change_columns(batch):
    batch["audio_id"] = batch["提示卡編號"] + "-" + batch["句編號"] + "-04.wav"
    batch["speaker_id"] = batch["發音人"][:2] + "_" + batch["發音人"][:3] + batch["發音人"][3:].zfill(4)
    batch["text"] = batch["漢羅台文"]
    return batch
tat = tat.map(change_columns, remove_columns=tat.column_names["train"], num_proc=4)

# convert to audio path, and remove audio id & speaker id
def get_file_path(batch):
    batch["path"] = os.path.join("wav", batch["speaker_id"], batch["audio_id"])
    return batch
tat = tat.map(get_file_path, remove_columns=["audio_id", "speaker_id"], num_proc=4)
print("Wav path: ", tat['train']['path'][0])

# load regex
with open('regex.txt', 'r', encoding='UTF-8') as f:
  lines = f.read()
print(lines)

# regex to re.sub format([\!])
bs = '\\'
regex = bs.join(lines)
rr = list(regex)
rr.insert(0, '[')
rr.append(']')
regex = ''.join(rr)

# remove regex
import re
def remove_regex(batch):
  batch["text"] = re.sub(regex, '', batch["text"])
  return batch
tat = tat.map(remove_regex)

"""
convert to numerical array will OOM, so run that function when fine-tune
"""

# split tarin and test, save tat.json as utf-8
train, test = tat['train'].train_test_split(.1).values()
train['train'].to_json("./train.json", encoding='utf8', force_ascii=False)
test['train'].to_json("./test.json", encoding='utf8', force_ascii=False)
print("dataset saved!")

"""### Create vocab.json"""
# collect unique character to dictionary
def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}
vocabs = tat.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=tat.column_names["train"])
vocab_list = list(vocabs["train"]["vocab"][0]) # vocab list[character]
vocab_dict = {v: k for k, v in enumerate(vocab_list)} # vocab dictionary{character: token id}

# add delimiter & unknown & padding token
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(len(vocab_dict))

# save vocab.json as utf-8
import json
with open('./output/vocab.json', 'w', encoding='utf8') as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)
print("vocab.json saved!")