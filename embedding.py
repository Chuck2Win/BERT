# -*- coding: utf-8 -*-

# position encoding
# attention is all you need
# sinusoidal
# pos : max_len과 관계, 각 element
# PE[pos,2i]=sin(pos/10000**((2i)/d_model)) <- i는 d_model의 element
# PE[pos,2i+1]=cos(pos/10000**((2i+1)/d_model)) <- i는 d_model의 element
import torch
import torch.nn as nn
import argparse


class positional_encoding(nn.Module):
    def __init__(self):
        pos = torch.arange(0,args.max_len).unsqueeze(1) # max_len, 1
        div = (10000**(torch.arange(0,args.d_model)/args.d_model)).unsqueeze(0) # 1, d_model
        self.pe = pos/div
        self.pe[:,0::2] = torch.sin(self.pe[:,0::2])
        self.pe[:,1::2] = torch.cos(self.pe[:,1::2])
        self.dropout = nn.Dropout(args.dropout)
    def forward(self, input):
        # input : (bs, seq_len, d_model) -> (bs, seq_len, d_model)
        output = input + self.pe[:args.seq_len,:].unsqueeze(0)        
        return self.dropout(output) # (max_len, d_model)        
    

input = torch.tensor([[1,1,1,1,1,2,2,2,2,2]])
segment_embedding = nn.Embedding(3,args.d_model,padding_idx=0)

class segment_embedding(nn.Module):
    # 0 
    # 1(first), 2(second)
    def __init__(self):
        self.segment_embedding = nn.Embedding(3,args.d_model,padding_idx=0)
    def forward(self,segment_input):
        # input : (bs, seq_len) - [1,1,1,1,1,1,2,2,2,2,2] 이런 식 -> (bs, seq_len, d_model)
        output = self.segment_embedding(segment_input)
        return output
    
class token_embedding(nn.Module):
    def __init__(self):
        self.token_embedding = nn.Embedding(args.n_vocab, args.d_model, padding_idx = args.padding_idx)
    def forward(self,input):
        # input : (bs, seq_len) -> (bs, seq_len, d_model)
        output = self.token_embedding(input) 
        return output
    

    
