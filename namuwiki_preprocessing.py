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

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default = './data/namuwiki.txt')
parser.add_argument("--output_data", type=str, default = './data/preprocessed_data_2.txt')

def substitute(i):
    try:
        return re.sub('[^ㄱ-ㅎ가-힇 ,.?0-9]+','',i)
    except:
        return ''
    
class preprocessing(object):
    def __init__(self):
        pass
    def transform(self):
        data = pd.read_csv(args.data,sep='\n',header=0,encoding='utf-8')
        # 한글,. ?숫자 외는 제거
        data['document'] = data['document'].apply(lambda i : substitute(i))
        new_data = data.loc[data['document']!='','document']
        new_data.to_csv(args.output_data,sep='\n',index=False,header=False)

if __name__=='__main__':
    args = parser.parse_args()
    p = preprocessing()
    p.transform()    