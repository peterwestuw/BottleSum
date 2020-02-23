#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:06:29 2019

@author: peterawest
"""

from itertools import combinations
import torch
from os import listdir
from random import shuffle
from nltk.tokenize import sent_tokenize
from sacremoses import MosesTokenizer, MosesDetokenizer
from gpt2_token_mod import gpt2_split, gpt2_join
from pytorch_pretrained_bert import GPT2Tokenizer

mt = MosesTokenizer()
mdt = MosesDetokenizer()    
punctuation = ".,:;\"\'"

global_tokenizer = None

def token_split(s, method = 'split', tokenizer = None):
    ''' Given a string s, tokenize '''
    if method == 'split':
        return s.split()
    if method == 'moses':
        tokenized_text = mt.tokenize(s, return_str=True)
        return tokenized_text.split()
    if method == 'gpt2':
        if tokenizer is None:
            global global_tokenizer
            if global_tokenizer is None:
                global_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer = global_tokenizer
        return gpt2_split(tokenizer, s)
    
    assert(False)
        
def token_join(list_s, method = 'split', s1 = None, autocap = True):
    ''' Given a token split version of s, produce a string
    corresponding to it. s1 is for options that need to look 
    at the string s1 that s is a substring of to decide how to
    reconstruct s'''
    list_s = list_s.copy()
    if autocap:
        
        ## get first non-punctuation word
        i = 0
        while (i < len(list_s)) and (list_s[i] in punctuation):
            i += 1
        if i < len(list_s) and (list_s[i][0] not in punctuation): # second condition prevents decapitalizing bug
            
            # capitalize first non-punctuation word if there is one
            
            # only if the word is not already capitalized
            if not list_s[i][0].isupper(): 
                
                list_s[i] = list_s[i].capitalize()
    
    if method == 'split': 
        return ''.join(word + ' ' for word in list_s)
    if method == 'moses':
        return mdt.detokenize(list_s,return_str=True)
    if method == 'gpt2':
        return gp2_join(list_s)
    
    assert(False)

def get_sublists(original, min_len = None, max_len = None, only_consec = False):
    ''' 
    Given original list generate all (not necessarily consecutive) 
    sublists of length min_len to max_len, and return this list of lists
    inputs:
        original is a list representing the full original list
        min_len is minimum sublist length
        max_len is maximum sublist length
    output:
        sublists is a list of all sublists that fit the desired length constraints
    '''
    
    if min_len is None:
        min_len = 1
    if max_len is None:
        max_len = len(original) - 1
        
    sublists = []
    
    
    if not only_consec: 
        for l in range(min_len, max_len + 1):
            sublists += list(combinations(original,l))
    else: # only allow consecutive elimination
        for l in range(min_len, max_len + 1):
            
            skip = len(original) - l # how many words to skip in this iteration
            for i in range(0,len(original)): # try deleting at every position
                if i + skip <= len(original): # only include this if we can delete 'skip' words at this i
                    sublists += [original[0:i] + original[i+skip:]] # add on this sublist
        
        pass
        
    return sublists


def get_padded_subseqs(S1, S2, pad_idx = 0, min_len = None, max_len = None, include_S1 = False):
    '''This function takes subsequences of S1 (as defined by min_len, max_len)
    padded to all be the same length. For instance, given 
    S1 = [1,2,3]
    S2 = [4,5]
    pad_idx = 0 
    this function should return a tensor like
    [[1,4,5,0],
     [2,4,5,0],
     [3,4,5,0],
     [1,2,4,5],
     [1,3,4,5]
     [2,3,4,5]]
    and a logit padding tensor like:
    [[1,1,0,0],
     ...
     [0,1,1,0]]
     
    inputs:
        S1: a list containing the sequence to be masked/subsequenced
        S2: a list containing the sequence that follows S1
        pad_idx: padding idx to use to pad these sequences
        min_len: minimum len for subsequences of S1
        max_len: minimum len for subsequences of S1
    outputs:
        padded_seqs: tensor of size N by (len(S1) + len(S2))
                    where N is the number of subsequences generated from S1. Row i 
                    contains the ith subsequences, followed by S2, then padding
        S2_mask: a tensor mask of size N by (len(S1) + len(S2))
                where N is as above, row i contains LM mask for CE of
                S2 tokens
    '''
    # return masked versions of S1
    sublists = get_sublists(S1, min_len = min_len, max_len = max_len)
    
    if include_S1:
        sublists += [S1]

    # padded_seqs is a list of padded lists of indices 
    padded_seqs = [list(S1_) + S2 + [pad_idx]*(len(S1) - len(S1_)) for S1_ in  sublists]

    # S2_mask will mask for the logits predicting S2 for an LM run on padded_seqs
    S2_mask = [[0]*(len(S1_) - 1) + [1]*len(S2) + [0]* (len(S1) - len(S1_) + 1) for S1_ in sublists]

    return torch.tensor(padded_seqs), torch.tensor(S2_mask) == 1

def test_subseqs(S1, S1_, S2, pad_idx = 0):
    '''
    Returns tensors analogous to get_padded_subseqs but with S1
    and S1_ as the only candidates for the first sentence
    '''

    sublists = [S1, S1_]


    # padded_seqs is a list of padded lists of indices 
    padded_seqs = [list(S1_) + S2 + [pad_idx]*(len(S1) - len(S1_)) for S1_ in  sublists]

    # S2_mask will mask for the logits predicting S2 for an LM run on padded_seqs
    S2_mask = [[0]* (len(S1_) - 1) + [1]*len(S2) + [0]* (len(S1) - len(S1_) + 1) for S1_ in sublists]

    return torch.tensor(padded_seqs), torch.tensor(S2_mask) == 1




####
# New functions for beam search code
####
    
def remove_words(S, rem_words, consecutive = False, tok_method = 'split', window = None):
    '''This function returns a list of sentences S1_list, which contains 
    all possible S1_ such that between 1 and rem_words are removed from S1'''
#    words2sent = lambda words: ''.join(word + ' ' for word in words)
    
    # S1 should already be a list of "words" or "tokens"
    #S1_words = token_split(S1, method = tok_method)
    
    if window is None:
        sublist_inds = get_sublists(list(range((len(S)))), min_len = max([len(S) - rem_words, 1]), max_len = len(S) - 1,
                                    only_consec = consecutive)
    else: 
        assert(window > 5) # window should be at least 5 words (preferably around 30)
        
        total_range = list(range(len(S))) # total range of inds
        window_range = total_range[:window] # those inds insisde the window
        tail = total_range[window:] # all inds outside the window
        
        
        ## only inds inside the window can be deleted
        sublist_inds_clipped = get_sublists(window_range, min_len = max([len(window_range) - rem_words, 1]), max_len = len(window_range) - 1,
                                    only_consec = consecutive)
        
        sublist_inds = [inds + tail for inds in sublist_inds_clipped] # add the tails back on
        
        
        

    # this is a list of lists of tokens (substrings)
    list_S1_ = [[S[i] for i in sublist_ind] for sublist_ind in sublist_inds]
    
    return list_S1_


def get_expansions(S1_list, rem_words, consecutive = False,tok_method = 'split', window = None):
    '''this runs remove_words on every sentence in S1_list and returns a list
    of all candidates (with duplicates removed)
    
    NOTE: S1_list should be a list of lists of substrings. In other words,
    each element of S1_list should be a list of strings that might be returned
    by the function token_split on some string'''
    
    if not consecutive:
        print('Warning: considering non-consecutive eliminations (time consuming)')
    
    expansions_list = []
    
    for S1_ in S1_list:
#        print(S1_)
        expansions_list += remove_words(S1_, rem_words, consecutive = consecutive,tok_method = tok_method, window = window)
    
    # remove duplicates (WARNING: you can still get duplicates in beam search
    # later but this is here for efficiency)
#    expansions_list = list(set(expansions_list))
    
    # remove duplicates
    as_tuple = [tuple(v) for v in expansions_list]  
    expansions_list = [list(v) for v in list(set(as_tuple))]

    return expansions_list
    

def get_padded_S1_S2(S1_list, S2, tokenizer, pad_idx = 0, bos = ' '): #, tok_method = 'split'):
    '''
    This takes a list of candidates for S1_ (a list of strings)
    '''

    S2_bpe = tokenizer.encode(S2)
    S1_bpe_prefix = tokenizer.encode(bos)
    
    assert(len(S1_bpe_prefix) == 1) # this should only be 1 bpe token 

    list_S1_ = S1_list

    # Add bpe prefix to each string, and encode each S1_ candidate.
    # note, we token_split and token_join each candidate to make sure 
    # its format is correct (may not directly map)

    list_S1_bpe = [S1_bpe_prefix + tokenizer.encode(S1_) for S1_ in list_S1_]
    
    list_S1_bpe_lens = [len(S1_) for S1_ in list_S1_bpe]
    list_total_seqs = [S1_ + S2_bpe for S1_ in list_S1_bpe]
    
    # max number of bpe tokens in a sequnce
    max_len = max(list_S1_bpe_lens) + len(S2_bpe)
    # total number of samples
    n_seqs = len(list_total_seqs)
    
    X = torch.ones([n_seqs, max_len])*pad_idx

    mask = torch.zeros([n_seqs, max_len])
    
    # mask is for CE, so offset by 1 from actual words (note this is for S2)
    for i in range(n_seqs):
        seq = list_total_seqs[i]
        len_S1_ = list_S1_bpe_lens[i]
        
        X[i,:len(seq)] = torch.tensor(seq)
        mask[i,(len_S1_ - 1):(len_S1_-1 + len(S2_bpe))] = 1
        
    return X.long(), mask




####
#
# Functions to deal with cnn data
#
####


def get_cnn_stories():
    file_paths = ['data/cnn/stories/' + file for file in listdir('data/cnn/stories/') if file.endswith('.story') and not file.startswith('._')]
    return file_paths
    
def get_cnn_paragraphs(story):

    lines = []
    
    with open(story,'r') as f:
        for line in f.readlines():
            
            # don't include the highlights
            if line.startswith('@highlight'):
                    break
            if len(line.split()) > 0: # if line is not functionally empty
                lines += [line.rstrip()]
    return lines

def get_cnn_sentences(story):
    
    # clean up document
    lines = []
    with open(story,'r') as f:
        for line in f.readlines():
            
            # don't include the highlights
            if line.startswith('@highlight'):
                    break
            if len(line.split()) > 0: # if line is not functionally empty
                lines += [line.rstrip()]
        
        doc = ''
        for line in lines:
            doc += line + ' '
        
        
        sents = sent_tokenize(doc)
    return sents


def get_cnn_split(split): # as in the original cnn dailymail paper
    if split == 'train':
        file_name = 'data/train_file_names.txt'
    elif split == 'val':
        file_name = 'data/val_file_names.txt'
    elif split == 'test':
        file_name = 'data/test_file_names.txt'
    else:
        assert(False)
    with open(file_name,'r') as f:
        lines = list(f.readlines())
    
    output = ['data/cnn/stories/' + line.strip() for line in lines]
    return output

CE_nored = torch.nn.CrossEntropyLoss(reduction='none') 

def get_CE_list(model, contexts, targets, back = False, batch=False, red = True):
    ''' 
    This gets the cross-entropy loss for the target sequences given the context sequences
    
    e.g. calling this with
    contexts = [X_0, X_1, X_2, X_3, ...]
    targets = [Y_0, Y_1, Y_2, Y_3, ...]
    
    returns 
    
    [-log( P(Y_0|X_0))  , -log( P(Y_1|X_1))   ] ...
    
    The purpose is to automatically handle a series of different length
    sequences of contexts and targets in large, batched matrices 
    
    Inputs:
        model: pytorch model to use (for now, should be a gpt2 version)
        contexts: list of lists of tokens [X_0, ... ,X_i, ... ] where X_i
          is the list of context tokens for context/target pair i
        targets: list of lists of tokens [Y_0, ... ,Y_i, ... ] where Y_i
          is the list of target tokens for context/target pair i
        back: whether or not the model is a forward or backward language
          model (this should typically be False)
        batch: False, or batch size
        red: whether to return a list of cross entropies (above) or a list 
          of lists (see reduction for torch CrossEntropyLoss for analogous options)
    
    '''    
            
    
    torch.cuda.empty_cache()
    device = model.transformer.wte.weight.device
    
    # contexts and targets should be same size!
    assert(len(contexts) == len(targets))
    
    
    
    if batch:
        # if batching, might sort by length to speed up transformer by not doing much padding
        
        
        ce_list = []
        while len(contexts) > 0:
            ce_list += get_CE_list(model, contexts[:batch], targets[:batch], back = back, batch=False, red=red)
            contexts = contexts[batch:]
            targets = targets[batch:]
        return ce_list
    
    
    # get number of sequences and max length
    n_seqs = len(contexts)
    max_len = max([len(context + target) for context,target in zip(contexts, targets) ] )
    
    
    # initialize inputs
    inp = torch.zeros((n_seqs, max_len)).long()
    
    target_mask = torch.zeros((n_seqs, max_len))
    
    # construct input matrix
    for i in range(n_seqs):
        context, target = contexts[i], targets[i]
        
        # reverse seqs if going backwards
        if back:
            context = context[::-1]
            target = target[::-1]
        
        # set values in input for this sequence
        inp[i, :len(context)] = torch.tensor(context).long()
        inp[i, len(context): len(context) + len(target) ] = torch.tensor(target).long()
        
        # mask is 1 where inp is the target
        target_mask[i, len(context): len(context) + len(target)] = 1
    
        
    logits = model(inp.to(device))[0]
    # trim last logit
    logits = logits[:,:-1]
    logits_shape = logits.shape
    
    targ = inp[:,1:].cpu().contiguous()
    targ_mask = target_mask[:,1:]
    
    #print(logits.cpu().view(-1).shape)
    #print(targ.view(-1).shape)
    ce_mat = CE_nored(logits.cpu().contiguous().view(-1, logits_shape[-1]), targ.view(-1))
    
    ce_mat = ce_mat.view(logits_shape[:-1])
    
    # if reducing (i.e. sum along target)
    if red:
        ce_list = (ce_mat*targ_mask).sum(dim=1)
        return ce_list.cpu().tolist()
    else:
        out_list = []
        for i in range(n_seqs):
            out_list += [ ce_mat[i, targ_mask[i,:] == 1].tolist()[::-1] ]
        return out_list
