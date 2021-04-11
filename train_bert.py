# -*- coding: utf-8 -*-

# train bert

import os
import argparse
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
#from model import *
import visdom
import torch.nn as nn
from tqdm import tqdm
from model import *
import transformers

def make_data_loader():
    F = open(args.total_data,'rb')
    ids = []
    segment_ids = []
    labels = []
    is_next = []
    #n = 0
    while True:
        try:
            p = pickle.load(F)
            ids.append(p[0])
            segment_ids.append(p[1])
            labels.append(p[2])
            is_next.append(p[3])
            # n+=1
            # if n==1000:
            #    break
        except:
            F.close()
            break
    
    
    X = TensorDataset(torch.LongTensor(ids),torch.LongTensor(segment_ids),torch.LongTensor(labels))#,torch.LongTensor(is_next))
    dataloader = DataLoader(X,args.batch_size,shuffle=True)
    return dataloader

class Config(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
args=Config({'batch_size':32,
 'epochs': 100,
 'lr':5*1e-5,
 'weight_decay':0.01,
 'betas':[0.9,0.999],
 'lr_warmup': 4000,
 'dropout':0.1,
 'n_layers':4,
 'tokenizer_file':'./tokenizer_model2',
 'n_head':4,
 'd_model': 128,
 'd_ff':4*128,
 
 'n_vocab':8000,
 'seq_len':128,
 'max_len':9999,
 'padding_idx':0,
 'total_data':'./data/total2.txt',
 'device':'cuda'})


#### train
device = args.device
model = BERT_pretrain(args).to(device)
criterion1 = nn.CrossEntropyLoss(ignore_index=args.padding_idx)
#criterion2 = nn.CrossEntropyLoss()
dataloader = make_data_loader()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
scheduler=transformers.get_linear_schedule_with_warmup(optimizer,args.lr_warmup,args.epochs*len(dataloader))
#### for visdom
mlm_loss = []
#nsp_loss = []
vis = visdom.Visdom()
# mlm loss
plot1 = vis.line(Y=torch.Tensor([0]),X=torch.Tensor([0]), opts = dict(title='mlm_loss',showlegend=True),env='bert')
# nsp loss
#plot2 = vis.line(Y=torch.Tensor([0]),X=torch.Tensor([0]), opts = dict(title='nsp_loss',showlegend=True),env='bert')
# mlm acc
plot3 = vis.line(Y=torch.Tensor([0]),X=torch.Tensor([0]), opts = dict(title='mlm_acc',showlegend=True),env='bert')
# nsp acc
#plot4 = vis.line(Y=torch.Tensor([0]),X=torch.Tensor([0]), opts = dict(title='nsp_acc',showlegend=True),env='bert')

############# train ###################
model.train()
for epoch in tqdm(range(1,args.epochs+1),desc=' epoch'):#,mininterval=1800):
    mlm_acc=0 # batch size, seq len, n_vocab
    nsp_acc=0 # batch size, 2
    mlm_total_loss=0.
    #nsp_total_loss=0.
    mlm_count=0
    #nsp_count=0
    for data in dataloader:
        optimizer.zero_grad()
        data = [i.to(device) for i in data]
        ids,segment_ids,labels = data
        mlm_output=model.forward(ids,segment_ids)
        loss = criterion1(mlm_output.transpose(1,2),labels)
        #loss2 = criterion2(nsp_output,is_next)
        #loss = loss1+loss2
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        loss.backward()
        optimizer.step()
        
        m=mlm_output.argmax(-1).reshape(-1)
        l=labels.reshape(-1)
        idx=(l!=0)
        mlm_acc+=((m[idx]==l[idx]).sum()).item()
        mlm_count+=(idx.sum()).item()
        #nsp_acc+=((nsp_output.argmax(-1)==is_next).sum()).item()
        #nsp_count+=len(is_next)
        mlm_total_loss+=loss.item()
        #nsp_total_loss+=loss2.item()
        scheduler.step()
    mlm_loss.append(mlm_total_loss/len(dataloader))
    #nsp_loss.append(nsp_total_loss/len(dataloader))
    vis.line(Y=torch.Tensor([mlm_total_loss/len(dataloader)]),X=torch.Tensor([epoch]),win=plot1, opts = dict(title='mlm_loss',showlegend=True),update = 'append',env='bert')
    #vis.line(Y=torch.Tensor([nsp_total_loss/len(dataloader)]),X=torch.Tensor([epoch]),win=plot2, opts = dict(title='nsp_loss',showlegend=True),update = 'append',env='bert')
    vis.line(Y=torch.Tensor([mlm_acc/mlm_count]),X=torch.Tensor([epoch]), win=plot3,opts = dict(title='mlm_acc',showlegend=True) ,update = 'append',env='bert')
    #vis.line(Y=torch.Tensor([nsp_acc/nsp_count]),X=torch.Tensor([epoch]),win=plot4,opts = dict(title='nsp_acc',showlegend=True), update = 'append',env='bert')
    
    if epoch%10==0:
        torch.save(model.state_dict(),'./epoch_%d'%epoch)
   