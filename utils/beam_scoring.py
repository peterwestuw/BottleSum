#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:57:42 2019

@author: peterawest
"""

import torch 

#from utils import get_padded_subseqs, test_subseqs
import utils.utils as utils
from untils.utils import get_CE_list
from pytorch_pretrained_bert import GPT2Tokenizer


flatten = lambda l : [y for x in l for y in x]

def masked_agg(ce, mask):   
    masked_ce = ce*mask[:,:-1].float()
    scores = masked_ce.sum(dim=1)
    return scores.cpu()

def get_scores_(X, model): # get unaggregated, word-by-word CE over X
    with torch.no_grad(): 

        X = X.to('cuda')

        X_in = X[:,:-1]
        X_target = X[:,1:]
        
        output, *_ = model(X_in,None)
#        output = output.contiguous() # bring back into standard format
        
        CE = torch.nn.CrossEntropyLoss(reduce = False)
        
        ce = CE(output.view(-1, output.shape[2]), X_target.contiguous().view(-1)).view(X_target.shape)

        scores = ce
        
        return scores


def get_CE_scores(S1_list, S2, tokenizer, model, max_tokens_batch, version = 0):
    if version == 0:
        return get_CE_scores_0(S1_list, S2, tokenizer, model, max_tokens_batch)
    elif version == 1:
        return get_CE_scores_1(S1_list, S2, tokenizer, model, max_tokens_batch)

def get_CE_scores_1(S1_list, S2, tokenizer, model, max_tokens_batch):
    
    # tokenize S1 sentences
    S1_list = [tokenizer.encode(S1) for S1 in S1_list]
    
    # S2 list just repeats S2
    S2 = tokenizer.encode(S2)
    S2_list = [S2]*len(S1_list)
    
    
    # to control for memory issues, limit the number of tokens-per-batch
    batch_size = int(max_tokens_batch/(max([len(S1) for S1 in S1_list]) + len( S2) ) )
    n_batches = int((len(S1_list) -1)/batch_size ) + 1
    
    # get conditional S2 scores 
    S2_scores = get_CE_list(model, S1_list, S2_list, batch=n_batches, red = True)
    
    # get unconditional S1 scores
    S1_Scores = get_CE_list(model, [tokenizer.encode(' ')]*len(S1_list), S1_list, batch=n_batches, red = True)
    
    
    return S1_scores, S2_scores
    

def get_CE_scores_0(S1_list, S2, tokenizer, model, max_tokens_batch): #, tok_method = 'split'):
    '''
    This funciton takes:
        S1_list: a list of candidate summaries of true S1
        S2: the sequence following true S1
        tokenizer: the tokenizer for the language model
        model: model used to calculate CE
        max_tokens_batch is the maximum number of tokens to be processed in 
            a batch (this should depend on gpu used)
    
    and calculates cross entropy (CE) for each S1_, and for S2 given each
    S1_ as a prefix.        
    '''

    X, mask = utils.get_padded_S1_S2(S1_list, S2, tokenizer, bos=' ') #,tok_method = tok_method)
    

    ## Entirety of every sample must fit in the context window 
    assert(X.shape[1] < 1024)
     
    batch_size = int(max_tokens_batch/X.shape[1])

    n_rounds = int((X.shape[0] -1)/batch_size ) + 1
    S1_score_list = []
    S2_score_list = []

    for j in range(n_rounds):
        
        X_batch = X[j*batch_size : (j+1)*batch_size]
        
        mask_S2 = mask[j*batch_size : (j+1)*batch_size ]
        
        # mask_S1 is everything before mask_S2, minus the first word (this should be some BOS token, from get_padded_S1_S2)
        mask_S1 = mask_S2.cumsum(dim=1) < 1

        scores = get_scores_(X_batch, model)
        
        mask_S1= mask_S1.to('cuda')
        mask_S2= mask_S2.to('cuda')
        S1_score_list.append(masked_agg(scores,mask_S1))
        S2_score_list.append(masked_agg(scores,mask_S2))

    S1_scores = torch.cat(S1_score_list, dim=0) 
    S2_scores = torch.cat(S2_score_list, dim=0) 


    return S1_scores, S2_scores
