# -*- coding: utf-8 -*-
"""## Training

Need to prepare train.json & test.json & vocab.json & audio
->
Output preprocessor & model

"""

"""### Load data"""
from datasets import load_dataset
train = load_dataset('json', data_files='./train.json')
test = load_dataset('json', data_files='./test.json')

# wav to array: {speech, sampling_rate, target_text}
import os
import soundfile as sf
datapath = './TAT/' # corpus root directory
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = sf.read(os.path.join(datapath, batch["path"]))
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]

    return batch
train = train.map(speech_file_to_array_fn, remove_columns=train.column_names["train"], num_proc=4)
test = test.map(speech_file_to_array_fn, remove_columns=test.column_names["train"], num_proc=4)

# check data type
import numpy as np
print("Target text:", train["train"][0]["target_text"])
print("Input array shape:", np.asarray(train["train"][0]["speech"]).shape)
print("Sampling rate:", train["train"][0]["sampling_rate"])
print("tat: ", train)

"""### Prepare trainer"""
# processor = feature_extractor + tokenizer
from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer("./output/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained("./wav2vec2-large-xlsr-tat/processor")
print("processor OK!")

# check that all files have the correct sampling rate
def prepare_dataset(batch):
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids

    return batch

train_prepared = train.map(prepare_dataset, remove_columns=train.column_names['train'], batch_size=8, num_proc=4, batched=True)
test_prepared = test.map(prepare_dataset, remove_columns=test.column_names['train'], batch_size=8, num_proc=4, batched=True)
print("tat_prepared: ", train_prepared)

# datacollator
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
print("collator OK!")

# metric
from datasets import load_metric
wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
print("metrics OK!")

# model
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
model.freeze_feature_extractor() # fine-tune
print("model OK!")

# training arguments: training 100 epochs, save model every 10 epochs
from transformers import TrainingArguments
training_args = TrainingArguments(
  num_train_epochs=100,
  group_by_length=True,
  fp16=True,
  prediction_loss_only=True,
  per_device_train_batch_size=14,
  per_device_eval_batch_size=4,
  dataloader_num_workers=4,
  learning_rate=3e-4,
  gradient_accumulation_steps=2,
  warmup_steps=500,
  output_dir="./wav2vec2-large-xlsr-tat/model",
  evaluation_strategy="steps",
  eval_steps=371,
  eval_accumulation_steps=100,
  save_strategy="steps",
  save_steps=3710,
  #save_total_limit=5,
  logging_dir="./wav2vec2-large-xlsr-tat/model/runs",
  logging_strategy="steps",
  logging_steps=371,
)
print("argument OK!")

# trainer
from transformers import Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_prepared['train'],
    eval_dataset=test_prepared['train'],
    tokenizer=processor.feature_extractor,
)
print("trainer OK!")

"""### Fine-tuning"""
# avoid OOM
import gc
gc.collect()
torch.cuda.empty_cache()
print("clean!")

# training
trainer.train()
print("done!")