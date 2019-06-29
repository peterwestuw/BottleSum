#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:47:36 2019

@author: peterawest
"""


import sys
import regex as re

def tokenize_words(self, text):
    """ Tokenize a string. """
    bpe_tokens = []
    bpe_words = []
    for token in re.findall(self.pat, text):
        if sys.version_info[0] == 2:
            token = ''.join(self.byte_encoder[ord(b)] for b in token)
        else:
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            
        bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        bpe_words.append(list(bpe_token for bpe_token in self.bpe(token).split(' ')))
    return bpe_tokens , bpe_words

def encode_words(self, text):
   tokens, words = tokenize_words(self,text)
   return self.convert_tokens_to_ids(tokens) , [self.convert_tokens_to_ids(word) for word in words]

# This is the one from the version you're using
def encode_old(self, text):
    bpe_tokens = []
    bpe_words = []
    for token in re.findall(self.pat, text):
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        
        bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        
        bpe_words.append(list(bpe_token for bpe_token in self.bpe(token).split(' ')))

    if len(bpe_tokens) > self.max_len:
        raise ValueError("Token indices sequence length is longer than the specified maximum  sequence length for this OpenAI GPT-2 model ({} > {}). Running this sequence through the model will result in indexing errors".format(len(bpe_tokens), self.max_len))

    
    return bpe_tokens , bpe_words#, [self.encode(word) for word in bpe_words]



def gpt2_split(tokenizer, text):
    A = encode_old(tokenizer, text)
    return [tokenizer.decode([tokenizer.encoder[a] for a in b]) for b in A[1]]

def gpt2_join(tokens):
    return ''.join(tokens)