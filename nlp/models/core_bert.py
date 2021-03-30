# reference:
# 
# https://huggingface.co/transformers/quickstart.html
# http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/
# https://www.youtube.com/watch?v=FKlPCK1uFrc&feature=youtu.be

import transformers
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
import random
import numpy as np

seed_val = 0
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def transformers_test():
    '''
    '''
    pass

# import logging
# logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = '[CLS] who was Jim ? [SEP]'
# text = '[CLS] who was Jim ? [SEP] Jim was a puppeteer [SEP]'
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens)

indexed_tokens = tokenizer.encode('who was Jim ?', add_special_tokens=True)
print(indexed_tokens)


segments_ids = [1, 1, 1, 1, 1, 1]
# segments_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensor = torch.tensor([segments_ids])

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
for n, p in model.named_parameters():
    if 'embedding' in n:
        print(n, p.shape)

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensor, attention_mask=None, output_all_encoded_layers=False)
    print(len(outputs))
    print(outputs[0].shape, outputs[1].shape)

    encoded_layers = outputs[0]


