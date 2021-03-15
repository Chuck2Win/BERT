# -*- coding: utf-8 -*-
# attention is all you need 에서 Transformer encoder만 따오면 된다.
# encoder layer = self attention layer(layer normalization+residual net) -> feed forward layer(layer normalization+residual net) 

# multi head 
import torch
import torch.nn as nn
import math

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    
args = Config({'d_model':128, 'max_len':128, 'n_head':8, 'd_k':128//8,'seq_len':32})

### embeddings ###
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

def gelu(x):
    '''
    gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.0044715x**3))
    '''
    return 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.0044715*x**3)))

# 각 layer에서 sample의 평균과 std를 구함(feature 무관)
# 그를 이용해서 각 sample를 정규화 시킴
# scaling and shifting - 이 것이 parameter임

class layer_norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1,args.seq_len,args.d_model)) # 1로 두는 까닭은 batch 마다 다를 필요가 없다.
        self.beta = nn.Parameter(torch.zeros(1,args.seq_len,args.d_model))
        self.eps = 1e-8
    def forward(self,input):
        # input shape : (bs,seq_len,d_model)
        mean = input.mean(-1,keepdim=True) # bs, seq_len,1
        std = input.std(-1,keepdim=True) # bs, seq_len,1
        output = (input-mean)/(std+self.eps) # bs, seq_len, d_model
        output = self.gamma*output+self.beta
        return output

class multi_head_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_Q = nn.Linear(args.d_model,args.d_model)
        self.linear_K = nn.Linear(args.d_model,args.d_model)
        self.linear_V = nn.Linear(args.d_model,args.d_model)
        
    def forward(self,input):
        # input (bs, seq_len, d_model) -> (bs,seq_len,h,d_k)
        Q = self.linear_Q(input) 
        Q = Q.reshape(-1,args.seq_len,args.n_head,args.d_k).transpose(1,2) # bs,h,seq_len,d_k
        K = self.linear_K(input) 
        K = K.reshape(-1,args.seq_len,args.n_head,args.d_k).transpose(1,2)
        V = self.linear_V(input) 
        V = V.reshape(-1,args.seq_len,args.n_head,args.d_k).transpose(1,2)
        
        softmax = nn.Softmax(3).forward(torch.matmul(Q,K.transpose(2,3))/math.sqrt(args.d_k))
        output = torch.matmul(softmax,V) # bs, h, seq_len, d_k
        output = output.transpose(1,2)
        output = output.reshape(-1,args.seq_len,args.n_head*args.d_k) # bs, seq_len, d_model
        return output

class feed_forward_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(args.d_model,args.d_ff)
        self.f2 = nn.Linear(args.d_ff,args.d_model)
        self.dropout = nn.Dropout(args.dropout)
    def forward(self,input):
        output = self.f1(input)
        output = self.dropout(gelu(output))
        output = self.f2(output)
        return output
    
class layer_connection(nn.Module):
    # input + dropout(layernorm(sublayer(input)))
    def __init__(self):
        super().__init__()
        self.layer_norm = layer_norm()
        self.dropout=nn.Dropout(args.dropout)
    def forward(self,input,sublayer):
        # input (bs, seq_len, d_model)
        # layer norm + dropout + residual net
        # attention is all you need 에선 , LayerNormalization(sublayer(input)+input)
        output = input + self.dropout(self.layer_norm(sublayer(input)))
        return output

class Transformer_Encoder_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_connection1 = layer_connection()
        self.mha = multi_head_attention()
        self.layer_connection2 = layer_connection()
        self.ffn = feed_forward_network()
    def forward(self,input):
        # multi head attention
        # feed forward network
        output1 = self.layer_connection1(input,self.mha)
        output2 = self.layer_connection2(output1,self.ffn)
        return output2
args['dropout']=0.1
args['d_ff']=2048
args['n_layers']=12
t = Transformer_Encoder_Layer()
output = t.forward(torch.randn((32,args.seq_len,args.d_model)))
encoder = nn.ModuleList([Transformer_Encoder_Layer() for _ in range(args.n_layers)])
o=encoder.forward()

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.TE = token_embedding()
        self.SE = segment_embedding()
        self.PE = positional_encoding()
        self.encoder = nn.ModuleList([Transformer_Encoder_Layer() for _ in range(args.n_layers)])
        self.linear = nn.Linear(args.d_model, args.n_vocab)
    def forward(self, input_ids, segment_ids, masks):
        # input ids (bs, seq_len) <- tokens
        # segment_ids (bs,seq_len) <- segments
        # masks (bs, seq_len) <- mask된 부분
        
        # embedding
        te = self.TE(input)
        se = self.SE(segment_ids)
        e = te+se
        output = self.PE(e)
        # encoder
        for i in range(args.n_layers):
            output = encoder[i](output)
        return output
        
    

    