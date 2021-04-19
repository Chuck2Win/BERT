# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:56:15 2021

@author: admin
"""

# preprocessing
# 필요한 text만을 추출
import argparse
import pandas as pd
import re

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default = './data/ratings_train.txt')
parser.add_argument("--output_data", type=str, default = './data/nsmc_preprocessed_data.txt')

def substitute(i):
    try:
        return re.sub('[^ㄱ-ㅎ가-힇 ,.?0-9]+','',i)
    except:
        return ''
        
class preprocessing(object):
    def __init__(self):
        pass
    def transform(self):
        data = pd.read_csv(args.data,sep='\t',header=0,encoding='utf-8')
        # 한글,. ?숫자 외는 제거
        data['document'] = data['document'].apply(lambda i : substitute(i))
        new_data = data.loc[data['document']!='','document']
        new_data.to_csv(args.output_data,sep='\n',index=False,header=False)
        return new_data


if __name__=='__main__':
    args = parser.parse_args()
    p = preprocessing()
    data=p.transform()    
    
