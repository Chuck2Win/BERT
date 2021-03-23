# -*- coding: utf-8 -*-
# data set
# [MASK] 씌우기
# NSP하기


# 데이터가 sentence로 들어오게 되면
# tokenizing 시키고
# 0.15의 비율로 [MASK] (디테일은 좀 다름)
# 50%의 비율로 is next or not next

# 해야할 일 
# 1. Wordpiece Tokenizer 만들고
# 2. Tokenizer로 쪼개고
import argparse
from transformers import BertTokenizer
# 3. [MASK]
# 4. NSP하기
import random
from collections import deque
import torch
import torch.nn as nn
import re
import pickle
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--corpus_file", type=str, default = 'D:/workspace/BERT/data/namuwiki.txt')
parser.add_argument("--tokenizer_file", type=str, default='D:/workspace/BERT/tokenizer_model') # 만들 Vocab의 숫자 


class Dataset:
    def __init__(self,args):
        
        self.data_path = args.corpus_file
        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_file,strip_accents=False,lowercase=False)
    
    def make_dataset(self): # 일단 MASK부터
        f = open(self.data_path,'r',encoding='utf-8')
        F1 = open('./masked_text.txt','wb')
        F2 = open('./masked_label.txt','wb')
        cnt = 0
        
        for _,line in tqdm(enumerate(f),desc='sentence',mininterval=60):
            data = f.readline()
            data=re.sub('\n','',data)
            cnt+=1
            sentence = self.tokenizer.encode(data, add_special_tokens = False) # tokenizer로 쪼개야함.
            new_data,label=self.masking(sentence)
            pickle.dump(new_data,F1)
            pickle.dump(label,F2)
            # p = random.random()    
            # if p>0.5: # is next
            #     new_sentence1,label1=self.masking(self.data[idx])
            #     new_sentence2,label2=self.masking(self.data[idx+1])
            #     isnext = 1#################################
                
            # else:
            #     new_sentence1,label1=self.masking(self.data[idx])
            #     index.pop(idx+1)
            #     rand = random.sample(list(range))
            #     if rand==(idx+1):
                    
            #     new_sentence2,label2=self.masking(self.data[idx+1])
            #     isnext = 1
    
    def isnext(self): # seq_len 
        pass
    
    def masking(self,sentence):
        # 이 때 sentence는 Wordpiece로 tokenizing된 데이터
        # sentence : (seq_len) - list 형태, 다 숫자화
        new_sentence=deque([])
        label = deque([]) # [MASK] 위치를 알아야한다. 즉 MLM을 실행할 장소를 알아야한다.
        for i in sentence: # O(seq_len) -> 근데 이것을 모든 sentence에 진행하게 되면.. 어마 무시할 듯 Nseqlen .
            p = random.random()
            if p<=0.15: # MASK
                s = random.random()
                if s<=0.8: # [MASK]
                    new_sentence.append(self.tokenizer.mask_token_id)
                elif 0.8<=s<0.9: # random
                    new_sentence.append(random.randint(0,self.tokenizer.vocab_size))
                else:
                    new_sentence.append(i)
                label.append(i)
            else:
                new_sentence.append(i)
                label.append(self.tokenizer.pad_token_id)
        return list(new_sentence),list(label)
            
if __name__ == '__main__':
    args = parser.parse_args()
    d = Dataset(args)
    d.make_dataset()
