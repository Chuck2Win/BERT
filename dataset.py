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
# 3. [MASK]
# 4. NSP하기
import argparse
from transformers import BertTokenizer
import random
from collections import deque
import torch
import torch.nn as nn
import re
import pickle
from tqdm import tqdm
import time
import torch
parser = argparse.ArgumentParser() 
parser.add_argument("--corpus_file", type=str, default = './data/preprocessed_data_2.txt') 
parser.add_argument("--tokenizer_file", type=str, default='./tokenizer_model2') 
parser.add_argument("--length", type=int, default= 1000000)
parser.add_argument("--seq_len", type=int, default= 128)
parser.add_argument("--output_file", type=str, default = './data/total2.txt') # {A_ids,A_labels,B_ids,B_labels,is_nest}
parser.add_argument("--mask_file", type=str, default = './data/masked_text2.txt')  
parser.add_argument("--label_file", type=str, default = './data/masked_label2.txt') 
        
class Dataset:
    def __init__(self):
        self.data_path = args.corpus_file
        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_file,strip_accents=False,lowercase=False)
        self.cnt = 0
        
    def make_mask(self):  
        f = open(self.data_path,'r',encoding='utf-8')
        F1 = open(args.mask_file,'wb')
        F2 = open(args.label_file,'wb')
        # NSP, MLM
        for _,line in tqdm(enumerate(f),desc='sentence',mininterval=60):
            data = f.readline()
            data=re.sub('\n','',data)
            sentence = self.tokenizer.encode(data, add_special_tokens=False) # tokenizer로 쪼개야함.
            if sentence: # 길이가 없는 것들도 존재한다. 그를 위해서.
                new_data,label=self.masking(sentence)
                # padding
                pickle.dump(new_data,F1)
                pickle.dump(label,F2)
                self.cnt+=1
            if self.cnt==args.length: # 너무 긴 파일의 경우.. 처리가 불가하기에.. 좋은 회사 가자!
                break
        f.close()
        F1.close()
        F2.close()
        
    def make_nsp_pairs(self): # 
        F1 = open(args.mask_file,'rb')
        F2 = open(args.label_file,'rb')
        F = open(args.output_file,'wb')
        
        cnt = 0
        candi_ids = None
        candi_label = None
        while True: # 시작과 끝에 대한 처리가 필요
            if candi_ids:    
                A_ids = candi_ids
                A_label = candi_label
            else:
                try:
                    A_ids = pickle.load(F1)
                    A_label = pickle.load(F2)
                except:
                    F.close()
                    break
            
            p = random.random()
            if p>0.5:
                # Not next
                is_next = 0
                F11 = open(args.mask_file,'rb')
                F22 = open(args.label_file,'rb')
                if self.cnt>10000:
                    a = list(range(10000))
                    if cnt<10000-1:
                        a.pop(cnt+1)
                    
                else:
                    a = list(range(self.cnt))
                    if cnt<self.cnt-1:
                        a.pop(cnt+1) # 다음 빼주기
                rn = random.sample(a,1)[0] # 맨끝 처리는 안해줌
                c=0
                while True:
                    try:
                        B_ids = pickle.load(F11)
                        B_label = pickle.load(F22)
                        if c>=rn:
                            break
                        else:
                            c+=1
                    except: # 맨 끝이였다는 뜻
                        B_ids = A_ids
                        B_label = A_label
                        break
                F11.close()
                F22.close()
                
            else:
                is_next = 1
                try:
                    candi_ids = pickle.load(F1)
                    candi_label = pickle.load(F2)
                    B_ids = candi_ids
                    B_label = candi_label
                except: # 마지막이란 소리
                    is_next = 0
                    B_ids = A_ids
                    B_label = A_label
                    F.close()
                    break
            # truncation+padding
            l = len(A_ids)+len(B_ids)
            if l>(args.seq_len-3):
                # A가 더 긴 경우 - 앞에서 부터 제거
                if len(A_ids)>=len(B_ids):
                    while (len(A_ids)+len(B_ids))>(args.seq_len-3):
                        A_ids.pop(0) 
                        A_label.pop(0)
                # B가 더 긴 경우 - 뒤에서 부터 제거
                else:
                    while (len(A_ids)+len(B_ids))>(args.seq_len-3):
                        B_ids.pop() 
                        B_label.pop()
                
            #print(len(A_ids))
            #print(len(B_ids))
                
            ids=[self.tokenizer.cls_token_id]+A_ids+[self.tokenizer.sep_token_id]+B_ids+[self.tokenizer.sep_token_id]+[self.tokenizer.pad_token_id]*(args.seq_len-3-(len(A_ids)+len(B_ids)))
            segment_ids = [1]*(len(A_ids)+2)+[2]*(len(B_ids)+1)+[0]*(args.seq_len - (len(A_ids)+len(B_ids)+3))
            
            labels=[self.tokenizer.pad_token_id]+A_label+[self.tokenizer.pad_token_id]+B_label+[self.tokenizer.pad_token_id]+[self.tokenizer.pad_token_id]*(args.seq_len-3-(len(A_ids)+len(B_ids)))
            
            
            #print(ids) 
            pickle.dump([ids,segment_ids,labels,is_next],F)    
            cnt+=1   
            if cnt == args.length:
                F.close()
                break
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
                    new_sentence.append(random.randint(0,self.tokenizer.vocab_size-1))
                else:
                    new_sentence.append(i)
                label.append(i)
            else:
                new_sentence.append(i)
                label.append(self.tokenizer.pad_token_id)
        return list(new_sentence),list(label)
            
if __name__ == '__main__': # 12,960,561 - 총 문장 개수
    now = time.time()
    args = parser.parse_args()
    d = Dataset()
    d.make_mask()
    d.make_nsp_pairs()
    print(time.time()-now)