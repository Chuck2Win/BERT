# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:56:15 2021

@author: admin
"""

# preprocessing
# 필요한 text만을 추출
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default = 'https://raw.githubusercontent.com/Chuck2Win/BERT/main/data/ratings_train.txt')
parser.add_argument("--output_data", type=str, default = './data/preprocessed_data.txt')

class preprocessing(object):
    def __init__(self):
        pass
    def transform(self):
        data = pd.read_csv(args.data,sep='\t',header=0,encoding='utf-8')
        # 딱히 전처리는 하지 않음
        data['document'].to_csv(args.output_data,sep='\n',index=False,header=False)
        


if __name__=='__main__':
    args = parser.parse_args()
    p = preprocessing()
    p.transform()    