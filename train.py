# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:02:35 2021

@author: admin
"""
from model import *
import torch
import torch.nn as nn
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("batch_size", type=int, help="batch size", default = 32)
parser.add_argument("lr", type=float, help="learning_rate", default = 1e-5)
parser.add_argument("dropout", type=float, help="dropout rate", default = 1e-1)
parser.add_argument("n_layers", type=int, help="number of encoder layers", default = 4)
parser.add_argument("n_head", type=int, help="number of heads", default = 4)
parser.add_argument("d_model", type=int, help="dimension of model", default = 128)
parser.add_argument("d_ff", type=int, help="dimension of ffn1", default = 128*4)
parser.add_argument("n_vocab", type=int, help="number of vocabs", default = 8000)
parser.add_argument("seq_len", type=int, help="length of sequence", default = 128)
parser.add_argument("max_len", type=int, help="maximum of positional encoding position parameter", default = 9999)


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    
args = Config({'d_model':128, 'max_len':128, 'n_head':8, 'd_k':128//8,'seq_len':32})
args['n_vocab']=8000
args['padding_idx']=1
args['batch_size']=8
args['dropout']=0.1
args['d_ff']=2048
args['n_layers']=12

input_ids = torch.randint(0,8000,(args.batch_size,args.seq_len))
segment_ids = torch.randint(1,3,(args.batch_size,args.seq_len))
a=BERT_pretrain(args)
x=a.forward(input_ids,segment_ids)
