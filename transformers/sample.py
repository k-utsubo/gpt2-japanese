#!/bin/env python
# encoding:utf-8
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Model, GPT2Config
from transformers import GPT2Tokenizer
#tokenizer=GPT2Tokenizer.from_pretrained("/Users/admin/data/gpt2ja-medium")
#output=tokenizer.tokenize("世界観が違う")
#print(output)


import torch
import sys
sys.path.append("..")
import encode_bpe
import json
with open('../ja-bpe.txt') as f:
    bpe = f.read().split('\n')

with open('../emoji.json') as f:
    emoji = json.loads(f.read())
encoder_ja=encode_bpe.BPEEncoder_ja(bpe,emoji)
inputs_ja={}
inputs_ja["input_ids"]=encoder_ja.encode("俺の名前は坂本俊之。何処にでもいるサラリーマンだ。")
model_ja = GPT2LMHeadModel.from_pretrained("/Users/admin/data/gpt2ja-medium")

outputs_ja = model_ja.generate(input_ids=torch.tensor([inputs_ja["input_ids"]]))
text_ja = encoder_ja.decode(
    outputs_ja[0].tolist())
print(text_ja)



from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("geekfeed/gpt2_ja")
model = GPT2LMHeadModel.from_pretrained("geekfeed/gpt2_ja")

inputs = tokenizer("俺の名前は坂本俊之。何処にでもいるサラリーマンだ。")
print(inputs)
outputs = model.generate(input_ids=torch.tensor([inputs["input_ids"]])
)
text = tokenizer.decode(
    outputs[0].tolist(),
#    skip_special_tokens=True,
#    clean_up_tokenization_spaces=True,
)
print(text)