# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:02:35 2021

@author: admin
"""
# 수정중..
from model import *
import torch
import pickle
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, help="batch size", default = 32)
parser.add_argument("--epochs", type=int, help="epochs", default = 40)
parser.add_argument("--lr", type=float, help="learning_rate", default = 1e-5)
parser.add_argument("--weight_decay", type=float, help="weight_decay", default = 0.01)
parser.add_argument("--betas", type=list, help="betas", default = [0.9,0.999])
parser.add_argument("--lr_warmup", type=int, help="learning rate warm up", default = 1000)
parser.add_argument("--dropout", type=float, help="dropout rate", default = 1e-1)
parser.add_argument("--n_layers", type=int, help="number of encoder layers", default = 4)
parser.add_argument("--tokenizer_file", type=str, default='./tokenizer_model') 
parser.add_argument("--n_head", type=int, help="number of heads", default = 4)
parser.add_argument("--d_model", type=int, help="dimension of model", default = 128)
parser.add_argument("--d_ff", type=int, help="dimension of ffn1", default = 128*4)
parser.add_argument("--n_vocab", type=int, help="number of vocabs", default = 8000)
parser.add_argument("--seq_len", type=int, help="length of sequence", default = 128)
parser.add_argument("--max_len", type=int, help="maximum of positional encoding position parameter", default = 9999)
parser.add_argument("--padding_idx", type=int, help="padding index", default = 0)
parser.add_argument('--total_data', type=str, help = 'preprocessed data - ids, segment_ids, labels, isnext', default = './data/total2.txt')
parser.add_argument('--device', type=str, help = 'device', default = 'cuda')

def make_data_loader():
    F = open(args.total_data,'rb')
    ids = []
    segment_ids = []
    labels = []
    is_next = []
    while True:
        try:
            p = pickle.load(F)
            ids.append(p[0])
            segment_ids.append(p[1])
            labels.append(p[2])
            is_next.append(p[3])
        except:
            F.close()
            break
    X = TensorDataset(torch.LongTensor(ids),torch.LongTensor(segment_ids),torch.LongTensor(labels))#,torch.LongTensor(is_next))
    dataloader = DataLoader(X,args.batch_size,shuffle=True)
    return dataloader

def train_pretrain():
    model.train()
    for epoch in tqdm(range(1,args.epochs+1),desc='epoch',mininterval=3600):
        for data in tqdm(dataloader,desc='steps',mininterval=600):
            optimizer.zero_grad()
            data = [i.to(device) for i in data]
            ids,segment_ids,labels = data#,is_next 
            mlm_output=model.forward(ids,segment_ids)
            loss1 = criterion1(mlm_output.transpose(1,2),labels)
            #loss2 = criterion2(nsp_output,is_next)
            loss = loss1#+loss2
            loss.backward()
            optimizer.step()
            scheduler.step()
        torch.save(model.state_dict(),'./%d_epoch'%(epoch))

if __name__ == '__main__':
    args = parser.parse_args()
    input_ids = torch.randint(0,8000,(args.batch_size,args.seq_len))
    device = args.device
    model = BERT_pretrain(args).to(device)
    criterion1 = nn.CrossEntropyLoss(ignore_index=args.padding_idx)
    #criterion2 = nn.CrossEntropyLoss()
    dataloader=make_data_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    scheduler=get_linear_schedule_with_warmup(optimizer,args.lr_warmup,args.epochs*len(dataloader))
    
    train_pretrain()
