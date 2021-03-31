# -*- coding: utf-8 -*-
# data set
# 해야할 일 
# 1. Wordpiece Tokenizer 만들고
# 2. Tokenizer로 쪼개고
import argparse
import os
from tokenizers import BertWordPieceTokenizer
parser = argparse.ArgumentParser()
parser.add_argument("--corpus_file", type=str, default = './data/preprocessed_data_2.txt')
parser.add_argument("--vocab_size", type=int, default=8000) # 만들 Vocab의 숫자 
parser.add_argument("--limit_alphabet", type=int, default=6000)
parser.add_argument("--min_freq", type=int, default=5)

if __name__=="__main__":
     args = parser.parse_args()
     tokenizer = BertWordPieceTokenizer(
                                   clean_text = True,
                                   handle_chinese_chars = True,
                                   strip_accents = False,
                                   lowercase = False,
                                   wordpieces_prefix='##'
                                   
                                   )
     tokenizer.train(
        files = [args.corpus_file],
        limit_alphabet=args.limit_alphabet,
        vocab_size = args.vocab_size,
        min_frequency = args.min_freq)
     if not os.path.isdir('./tokenizer_model2'):
         os.mkdir('./tokenizer_model2')
     tokenizer.save_model('./tokenizer_model2')
      

