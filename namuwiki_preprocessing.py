# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:03:27 2021

@author: admin
"""
import re
import pickle
import argparse
from tqdm import tqdm

# 5만개의 나무위키 문서들을 가지고 분석을 진행할 예정
# 각각의 문서마다 pickle 파일 만들기
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default = './data/namuwiki/namuwiki.txt')
parser.add_argument("--output_data", type=str, default = './data/namuwiki/namuwiki_document')

parser.add_argument("--document_count", type=int, default = 50000)

if __name__=='__main__':
    args = parser.parse_args()
    F = open(args.data,'r',encoding='utf-8')
    cnt = 0
    
    while True:
        doc = []
        while True:
            a = F.readline()
            if a=='\n': # 문서의 구분
                break
            
            
        if doc:
            cnt+=1
            f = open(args.output_data+'_%d'%cnt,'wb')
            pickle.dump(doc,f)
            f.close()
            
        if cnt==args.document_count:
            break
    
    
