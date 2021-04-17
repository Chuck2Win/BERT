# -*- coding: utf-8 -*-

# train bert

import os
import argparse
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
import visdom
import torch.nn as nn
from tqdm import tqdm
from model import *
import transformers
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import json
writer = SummaryWriter('runs/BERT')
## 명령줄
# tensorboard --logdir=runs
# https://localhost:6006
parser = argparse.ArgumentParser() 


parser.add_argument("--tokenizer_file", type=str, default='./tokenizer_model') 
parser.add_argument("--length", type=int, default= 20000)
parser.add_argument("--seq_len", type=int, default= 128) 
parser.add_argument("--input_file", type=str, default = './data/namuwiki_pretrain/') # {A_ids,A_labels,B_ids,B_labels,is_nest}
parser.add_argument("--batch_size",type = int, default = 32)
parser.add_argument("--epochs",type = int, default = 20)
parser.add_argument("--lr",type = float, default = 1e-3)
parser.add_argument("--weight_decay",type = float, default = 1e-2)
parser.add_argument("--betas",type = list, default = [0.9,0.999])
parser.add_argument("--lr_warmup",type = int, default = 10000)
parser.add_argument("--dropout",type = float, default = 1e-1)
parser.add_argument("--n_layers",type = int, default = 4)
parser.add_argument("--n_head",type = int, default = 4)
parser.add_argument("--d_model",type = int, default = 64)
parser.add_argument("--d_ff",type = int, default = 4*64)
parser.add_argument("--n_vocab",type = int, default = 16000)
parser.add_argument("--max_len",type = int, default = 9999)
parser.add_argument("--padding_idx",type = int, default = 0)
parser.add_argument("--device",type = str, default = 'cuda')

def make_data_loader():
    Input_ids = []
    Segment_ids = []
    Labels = []
    Is_next = []
    
    for i in tqdm(range(args.length)):
        with open(args.input_file+'%d.json'%i,'r') as f:

            data = json.load(f)

        Input_ids.append(data['input_ids'])
        Segment_ids.append(data['segment_ids'])
        Labels.append(data['labels'])
        Is_next.append(data['is_next'])
        
    X = TensorDataset(torch.LongTensor(Input_ids),torch.LongTensor(Segment_ids),torch.LongTensor(Labels),torch.LongTensor(Is_next))
    dataloader = DataLoader(X,args.batch_size,shuffle=True)
    return dataloader




if __name__ == '__main__':
    args = parser.parse_args()
    #### train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERT_pretrain(args).to(device)
    criterion1 = nn.CrossEntropyLoss(ignore_index=args.padding_idx,reduction='sum')
    criterion2 = nn.CrossEntropyLoss()
    dataloader = make_data_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,args.lr_warmup,args.epochs*len(dataloader))
    mlm_total_loss_check = []
    mlm_total_acc_check=[]
    nsp_total_loss_check = []
    nsp_acc_loss_check = []
    ############# train ###################
    model.train()
    for epoch in tqdm(range(1,args.epochs+1),desc=' epoch'):#,mininterval=1800):
        mlm_total_acc=0 # batch size, seq len, n_vocab
        nsp_total_acc=0 # batch size, 2
        mlm_total_loss=0.
        nsp_total_loss=0.
        nsp_count = 0
        n = 0
        for data in dataloader:
            optimizer.zero_grad()
            data = [i.to(device) for i in data]
            ids, segment_ids, labels, is_next = data
            mlm_output,nsp_output = model.forward(ids,segment_ids)
            loss1 = criterion1(mlm_output.transpose(1,2), labels)
            loss2 = criterion2(nsp_output,is_next)
            loss = loss1+loss2
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            m=mlm_output.argmax(-1).reshape(-1)
            l=labels.reshape(-1)
            idx=(l!=args.padding_idx)
            n+=((idx.sum()).item())
            mlm_total_acc+=(((m[idx]==l[idx]).sum()).item())
            nsp_total_acc+=((nsp_output.argmax(-1)==is_next).sum()).item()
            nsp_count+=len(is_next)
            mlm_total_loss+=(loss.item()) # mlm당
            nsp_total_loss+=(loss2.item())
        mlm_total_loss_check.append(mlm_total_loss/n)
        mlm_total_acc_check.append(mlm_total_acc/n)
        nsp_total_loss_check.append(nsp_total_loss/nsp_count)
        nsp_total_loss_check.append(nsp_total_acc/nsp_count)
        
        print(epoch)
        print('mlm_loss : %.3f'%(mlm_total_loss/n))
        print('mlm_acc : %.2f'%(mlm_total_acc/n))
        print('nsp_loss : %.3f'%(nsp_total_loss/nsp_count))
        print('nsp_acc : %.2f'%(nsp_total_acc/nsp_count))
        if epoch%5==0:
            torch.save(model.state_dict(),'./epoch_%d'%epoch)
    

