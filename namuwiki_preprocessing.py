# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:03:27 2021

@author: admin
"""
import re
import os
import pickle
import argparse
import pandas as pd


# 길이가 32 이상의 것으로 하고, 100만개만 뽑아내자
# 한글 숫자만 살리기
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default = './data/namuwiki.txt')
parser.add_argument("--output_data", type=str, default = './data/preprocessed_data_namuwiki.txt')
parser.add_argument("--sentence_count", type=int, default = 1000000)

def substitute(i):
    try:
        return re.sub('[^ㄱ-ㅎ가-힇 ,.?0-9]+','',i)
    except:
        return ''
    
if __name__=='__main__':
    args = parser.parse_args()
    F = open(args.data,'r',encoding='utf-8')
    f = open(args.output_data,'w',encoding='utf-8')
    n = 0
    while True:
        i = F.readline()
        i = re.sub('[^ㄱ-ㅎ가-힇 ,.?0-9]+','',i)
        if len(i)>32:
            f.write(i+'\n')
            n+=1
        if n==args.sentence_count:
            F.close()
            f.close()
            break
        
