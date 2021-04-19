# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:39:05 2021

@author: admin
"""

import numpy as np
import pandas as pd
from transformers import BertTokenizer
from model import *
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('./tokenizer_model',strip_accents=False,lowercase=False)

train_data = pd.read_csv('./data/ratings_train.txt',sep='\t', header=0, index_col = 0)
train_data = train_data.dropna()

test_data = pd.read_csv('./data/ratings_test.txt', sep = '\t', header = 0, index_col = 0)
test_data = test_data.dropna()
# tokenize + special token + Trunc, Pad

# train data
train_data['tokenized'] = train_data['document'].apply(lambda i : tokenizer.encode(i, add_special_tokens = True, truncation = True, padding = 'max_length', max_length = 128))
# test data
test_data['tokenized'] = test_data['document'].apply(lambda i : tokenizer.encode(i, add_special_tokens = True, truncation = True, padding = 'max_length', max_length = 128))

# train loader
X = TensorDataset(torch.LongTensor(train_data['tokenized'].tolist()), torch.LongTensor(train_data['label'].tolist()))
train_loader = DataLoader(X, batch_size = 8)

# test loader
X = TensorDataset(torch.LongTensor(test_data['tokenized'].tolist()), torch.LongTensor(test_data['label'].tolist()))
test_loader = DataLoader(X, batch_size = 8)

# configuration
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
import json
f = open('./config','r')
args = json.load(f)
args = Config(args)
# model load

model = BERT_pretrain(args)
model.load_state_dict(torch.load('./model/epoch_100'))
bert = model.bert



class classification_model(nn.Module):
    def __init__(self, bert, C):
        super().__init__()
        self.bert = bert
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(args.d_model, C)
    
    def forward(self,input):
        # input data
        # token_ids
        # segment ids 가 필요
        segment_ids = torch.ones_like(input, device = input.device)
        output = self.bert.forward(input, segment_ids)
        # [cls]만을 활용
        output = output[:,0,:] # bs, d_model
        output = self.activation.forward(output)
        out = self.classifier.forward(output)
        return out

# epoch 10
# Adam 1e-3
# betas : [0.9,0.999]
# weight decay : [0.01]

# train
model = classification_model(bert, 2).to(args.device)
epochs = 5
lr = 1e-4
betas = [0.9, 0.999]
weight_decay = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = lr, eps = 1e-8, weight_decay = weight_decay, betas = betas)
cost = []
total_acc = []
criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(1,epochs+1),desc='epoch'):
    model.train()
    ACC = 0
    Cost = 0
    for data in tqdm(train_loader,desc = 'step', mininterval = 60):
        optimizer.zero_grad()
        data = tuple(i.to(args.device) for i in data)
        output = model.forward(data[0])
        #outputss.append(output)
        loss = criterion(output, data[1])
        loss.backward()
        optimizer.step()
        Cost+=loss.item()
        out = output.argmax(-1)
        acc = (out==data[1]).float().mean()
        ACC+=acc.item()
    print(epoch)
    print(Cost/len(train_loader))
    print(ACC/len(train_loader))
    cost.append(Cost/len(train_loader))
    total_acc.append(ACC/len(train_loader))


############################################ test ########################################
with torch.no_grad():
    model.eval()
    ACC = 0
    Cost = 0
    for data in test_loader:
        data = tuple(i.to(args.device) for i in data)
        output = model.forward(data[0])
        #outputss.append(output)
        loss = criterion(output, data[1])
        Cost+=loss.item()
        out = output.argmax(-1)
        acc = (out==data[1]).float().mean()
        ACC+=acc.item()
    print(Cost/len(test_loader))
    print(ACC/len(test_loader))
    