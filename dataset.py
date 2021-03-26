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
parser = argparse.ArgumentParser() 
parser.add_argument("--corpus_file", type=str, default = './data/namuwiki.txt') # 1200만개의 데이터임 - 100만개만 활용하자!
parser.add_argument("--tokenizer_file", type=str, default='./tokenizer_model') # 만들 Vocab의 숫자 
parser.add_argument("--length", type=int, default=1000000) # 만들 Vocab의 숫자 
parser.add_argument("--output_file", type=str, default = './data/total2.txt') # {A_ids,A_labels,B_ids,B_labels,is_nest}
parser.add_argument("--mask_file", type=str, default = './data/masked_text.txt') # 만들 Vocab의 숫자 
parser.add_argument("--label_file", type=str, default = './data/masked_label.txt') # 만들 Vocab의 숫자 
        
class Dataset:
    
    def __init__(self):
        self.data_path = args.corpus_file
        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_file,strip_accents=False,lowercase=False)
        self.cnt = 0
        
    def make_mask(self): # 
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
                pickle.dump(new_data,F1)
                pickle.dump(label,F2)
                self.cnt+=1
            if self.cnt==args.length:
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
                # a = list(range(self.cnt)) # 너무 커서 10000개로 하자
                if args.length>10000:
                    a = list(range(10000))
                    if cnt<10000-1:
                        a.pop(cnt+1)
                    
                else:
                    a = list(range(args.length))
                    if cnt<args.length-1:
                        a.pop(cnt+1) # 다음 빼주기
                rn = random.sample(a,1)[0]
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
            pickle.dump({'A_ids':A_ids,'A_label':A_label,'B_ids':B_ids,'B_label':B_label,'is_next':is_next},F)    
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
                    new_sentence.append(random.randint(0,self.tokenizer.vocab_size))
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
    
    F1 = open(args.output_file,'rb')
#     F1 = open(args.corpus_file,'r',encoding='utf-8')
#     F1 = open('./data/masked_label.txt','rb')
    n = 0
    while True:
        try:
            pickle.load(F1)
            n+=1
            if n%(10**6)==0:
                print(n)
        except:
            F1.close()
            break
    
print(n)    

#     d = Dataset()
#     d.cnt = 12707495
#     d.make_nsp_pairs()
#     print(time.time()-now)
#     d.make_dataset()

# #############################################################

# F2 = open('./masked_label.txt','rb')
# F = open('./total.txt','rb')
# cnt = 0
# candi_ids = None
# candi_label = None

# while True: # 시작과 끝에 대한 처리가 필요
#     if candi_ids:    
#         A_ids = candi_ids
#         A_label = candi_label
#     else:
#         if cnt==5:
#             break
#         else:
#             A_ids = pickle.load(F1)
#             A_label = pickle.load(F2)
        
#     p = random.random()
#     if p>0.5:
#         print('this')
#         # Not next
#         is_next = 0
#         F11 = open('./masked_text.txt','rb')
#         F22 = open('./masked_label.txt','rb')
#         a = list(range(12707495))
#         if cnt<12707495:
#             a.pop(cnt+1) # 다음 빼주기
#         rn = random.sample(a,1)[0]
#         print(cnt)
#         print(rn)
#         c=0
#         while True:
#             try:
#                 B_ids = pickle.load(F11)
#                 B_label = pickle.load(F22)
#                 if c==rn:
#                     break
#                 else:
#                     c+=1
#             except: # 맨 끝이였다는 뜻
#                 B_ids = A_ids
#                 B_label = A_label
#                 break
#         F11.close()
#         F22.close()
        
#     else:
#         print('that')
#         is_next = 1
#         try:
#             candi_ids = pickle.load(F1)
#             candi_label = pickle.load(F2)
#             B_ids = candi_ids
#             B_label = candi_label
            
#         except: # 마지막이란 소리
#             is_next = 0
#             B_ids = A_ids
#             B_label = A_label
#             break
#     print(A_ids)
#     print(B_ids)
#     #print({'A_ids':A_ids,'A_label':A_label,'B_ids':B_ids,'B_label':B_label,'is_next':is_next})
#     #pickle.dump({'A_ids':A_ids,'A_label':A_label,'B_ids':B_ids,'B_label':B_label,'is_next':is_next},F)    
#     cnt+=1   
# F.close()
# F1.close()
# F2.close()






# #############################################################
# rn = random.sample(range(2),1)

# # test test
# # dataframe으로 만들어볼까?
# F1 = open('./masked_text2.txt','rb')
# F2 = open('./masked_label2.txt','rb')
# from pandas import DataFrame as df
# data = df()
# data['ids']=[0]*12707495
# data['labels']=None
# cnt=0
# while True:
#     try :
#         d1 = pickle.load(F1)
#         d2 = pickle.load(F2)
#         data['ids'][cnt]=d1
#         data['labels'][cnt]=d2
#         cnt+=1
#     except:
#         break
    
# F1.close()
# F2.close()

# cnt

# for _,i in enumerate(F1):
#     pickle.load(F1)
    
# pickle.load(F2)

# for _,i in enumerate(f1):
#     print(_)
#     f1.readline()
# f1.close()
# f2.close()
# f = open('D:/workspace/BERT/data/namuwiki.txt','r',encoding='utf-8')
# f.close()
# data = f.readline()
# data = re.sub('\n','',data)
# print(data)
# tokenizer = BertTokenizer.from_pretrained('D:/workspace/BERT/tokenizer_model',strip_accents=False,lowercase=False)
# tokenizer.mask_token_id
# data = '야이놈의 시끼야'
# sentence = tokenizer.encode(data,add_special_tokens=False,truncation = True, max_length = args.max_length, padding = 'max_length')
# sentence = tokenizer.encode(data,add_special_tokens=False,truncation = True, max_length = 128, padding = 'max_length')
# sentence
# from matplotlib import pyplot as plt
# plt.hist(lengths)
# from collections import deque
# new_sentence=deque([])
# label = deque([]) # [MASK] 위치를 알아야한다. 즉 MLM을 실행할 장소를 알아야한다.
# for i in sentence: # O(seq_len) -> 근데 이것을 모든 sentence에 진행하게 되면.. 어마 무시할 듯 Nseqlen .
#     p = random.random()
#     if p<=0.15: # MASK
#         s = random.random()
#         if s<=0.8: # [MASK]
#             new_sentence.append(self.tokenizer.mask_token_id)
#         elif 0.8<=s<0.9: # random
#             new_sentence.append(random.randint(0,self.tokenizer.vocab_size))
#         else:
#             new_sentence.append(i)
#         label.append(i)
#     else:
#         new_sentence.append(i)
#         label.append(self.tokenizer.pad_token_id)
# return list(new_sentence),list(label)