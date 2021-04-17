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
import pickle
import json
from tqdm import tqdm
parser = argparse.ArgumentParser() 

parser.add_argument("--docs", type=str, default = './data/namuwiki/namuwiki_document') 
parser.add_argument("--tokenizer_file", type=str, default='./tokenizer_model') 
parser.add_argument("--length", type=int, default= 20000)
parser.add_argument("--seq_len", type=int, default= 128) 
parser.add_argument("--output_file", type=str, default = './data/namuwiki_pretrain/') 


# Mask + NSP
      
# Mask + NSP

class pretrain_instances(object):
    def __init__(self,args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_file,strip_accents=False,lowercase=False)
        
    def mask(self, sentence):
        # 이 때 sentence는 Wordpiece로 tokenizing된 데이터
        # sentence : (seq_len) - list 형태, 다 숫자화
        new_sentence= []
        label = [] # [MASK] 위치를 알아야한다. 즉 MLM을 실행할 장소를 알아야한다.
        for i in sentence: # O(seq_len) -> 근데 이것을 모든 sentence에 진행하게 되면.. 어마 무시할 듯 Nseqlen .
            p = random.random()
            if p<=0.15: # MASK
                s = random.random()
                if s<=0.8: # [MASK]
                    new_sentence.append(self.tokenizer.mask_token_id)
                elif s<0.9: # random
                    x = self.tokenizer.pad_token_id
                    while x == self.tokenizer.pad_token_id: 
                        x = random.randrange(self.tokenizer.vocab_size)
                    new_sentence.append(x)
                else:
                    new_sentence.append(i)
                label.append(i)
            else:
                new_sentence.append(i)
                label.append(self.tokenizer.pad_token_id)
        return new_sentence, label 
    
    def trunc(self,segment_a, segment_b):
        while True:
            if len(segment_a) + len(segment_b) > (args.seq_len-3):
                if len(segment_a)>len(segment_b):
                    segment_a = segment_a[1:]
                else:
                    segment_b = segment_b[:-1]
            else:
                break
        
        return segment_a,segment_b 
    
    def pad(self,segment_a, segment_b):
        return segment_a+segment_b+[self.tokenizer.pad_token_id]*(self.args.seq_len-(len(segment_a)+len(segment_b))) 
        
    def make(self,doc_idx):
        '''
        docs는 전체 문서 e.g 나무위키
        doc_idx는 그 중에서 해당 문서 e.g 머신러닝의 idx 
        '''
        # encoding 시킴
        docs = self.args.docs
        f = open(docs+'_%d'%doc_idx,'rb')
        doc = pickle.load(f)
        f.close()
        new_doc = []
        for i in doc:
            new_doc.append(self.tokenizer.encode(i, add_special_tokens=False)) # tokenizer로 쪼개야함.)
        doc = new_doc
        segment = []
        length = 0
        is_next = 0
        if random.random()>0.5:
            # is_next
            is_next = 1
        
        for i in range(len(doc)): # 머신러닝 문서의 문장들
            segment.append(doc[i]) # doc[i] e.g [1,2,3,4,1]
            length+=len(segment)
            if i==(len(doc)-1):
                # case 1 . 문서의 마지막 문장 + 길이 512 이상
                if length>=args.seq_len:
                    # case 1.1 문서의 개수가 1개인 경우 - 데이터 버림
                    #if len(segment)==1:
                    # case 1.2 문서의 개수가 2개 이상인 경우
                    if len(segment)>=2:
                        # -1을 해주는 이유는 마지막 문장까지 포함하면 512 이상이 되므로
                        a_idx = random.randrange(len(segment)-1) 
                # case 2 . 문서의 마지막 문장 + 길이 512 미만
                else:
                    # -1을 해주는 이유는 마지막 문장까지 포함하면 b는 남는것이 없으므로
                    a_idx = random.randrange(len(segment)-1)
            # case 3. 문서의 마지막 문장 x 길이 512 이상
            else:
                if length>=args.seq_len:
                    if is_next==1:
                    # is next면 b에도 segment의 일정부분을 가져가니깐
                    # -1 해주는 까닭은 마지막 문장까지 포함하면 512 이상이 되므로
                        a_idx = random.randrange(len(segment)-1)
                    else:
                    # 그게 아니면 끝까지 a가 써도 된다
                        a_idx = random.randrange(len(segment))
                    break
        #☆ print(a_idx)
        # segment a 완성시키기
        segment_a=[]
        for i in range(a_idx+1):
            segment_a.extend(segment[i])
        segment_a+=[]    

        # segment b 만들기
        segment_b = []
        # is_next라면 a_idx~len(segment)에서 문장을 뽑는다
        if is_next==1:
            for i in range(a_idx+1,len(segment)):
                segment_b.extend(segment[i])
        else:
            # 다른 문서에서 문장을 추출해야함.
            random_doc_idx = doc_idx
            # doc_idx와 다를때까지 sampling - 멋진 표현이네
            while random_doc_idx == doc_idx:
                random_doc_idx = random.randrange(len(docs))
            f = open(docs+'_%d'%random_doc_idx,'rb')
            random_doc = pickle.load(f)
            f.close()    
            new_random_doc = []
            for i in random_doc:
                new_random_doc.append(self.tokenizer.encode(i, add_special_tokens=False)) # tokenizer로 쪼개야함.)
            random_doc = new_random_doc
            # start를 자유롭게 시작 
            # -1 해주는 까닭은 마지막 문장으로 되면
            # 공란으로 되어버림
            random_start = random.randrange(0,len(random_doc)-1)
            for j in range(random_start, len(random_doc)):
                segment_b.extend(random_doc[j])

        # trunc : a는 앞에서부터, b는 뒤에서 부터
        # mask  
        # a에는 cls token, sep token, b에는 sep token 
        # pad 
        segment_a, segment_b = self.trunc(segment_a,segment_b)
        masked_segment_a, masked_label_a = self.mask(segment_a)
        masked_segment_b, masked_label_b = self.mask(segment_b)
        masked_segment_a = [self.tokenizer.cls_token_id]+masked_segment_a+[self.tokenizer.cls_token_id]
        masked_label_a = [self.tokenizer.pad_token_id]+masked_label_a+[self.tokenizer.pad_token_id]
        masked_segment_b = masked_segment_b+[self.tokenizer.cls_token_id]
        masked_label_b = masked_label_b+[self.tokenizer.pad_token_id]
        
        input_ids = masked_segment_a+masked_segment_b
        segment_ids = [1]*len(masked_segment_a)+[2]*len(masked_segment_b)
        labels = masked_label_a+masked_label_b
        if len(input_ids)<self.args.seq_len:
            input_ids += ([self.tokenizer.pad_token_id]*(self.args.seq_len-len(input_ids)))
            segment_ids += ([0]*(self.args.seq_len-len(segment_ids)))    
            labels += ([self.tokenizer.pad_token_id]*(self.args.seq_len-len(labels)))
        instance = {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'labels': labels,
            'is_next': is_next
        }           
        return instance
    def make_docs(self):
        # json으로 저장
        for i in tqdm(range(self.args.length),desc='docs'):
            #print(i) # ★
            instance = self.make(i)
            with open(args.output_file+'%d.json'%i,'w') as f:
                json.dump(instance,f)
        

    
if __name__ == '__main__': # 12,960,561 - 총 문장 개수
    args = parser.parse_args()
    
    d = pretrain_instances(args)
    d.make_docs()


