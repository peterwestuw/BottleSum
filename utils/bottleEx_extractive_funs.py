#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:14:12 2019

@author: peterawest
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:04:05 2019

@author: peterawest

v1 aims to be more efficient by calculating scores AFTER culling duplicates

v5 aims to simplify by adding the very simple filter condition (must just have 
lower )

"""


from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel

from utils.beam_scoring import get_CE_scores

#from utils import get_padded_subseqs, test_subseqs
import utils.utils as utils



# flatten a 1-level nested list
flatten = lambda l : [y for x in l for y in x]

def elimination_beam_search(S1, S2, # input sentences
                            k = 1, 
                            rem_words = 1, 
                            tok_method = 'moses', 
                            autocap = True,
                            model_path = '',
                            max_tokens_batch = 20000, # based on GPU memory, for efficiency
                            model = None,
                            tokenizer = None,
                            min_words = 1, # minimum generation length
                            window = None, # window in which to consider deletions (for long sentence efficiency)
                            parent_version = 'min'):
    
    '''
    Input:
        
    '''
    

    
    

    assert(min_words > 0)
    

    #parent_version = 'min' # must be less than min parent
    parent_CE = {}
    parent_len = {}

    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    
    if model is None:
        if model_path == '':
            model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
        else: # use a fine-tuned version
            model = GPT2LMHeadModel.from_pretrained(model_path).to('cuda')
        model.eval()
    max_tokens_batch = max_tokens_batch


    ##
    # Define stuff needed for scoring function(s)
    ##
    
    S1split = utils.token_split(S1, method =tok_method)

    # make sure cannonical S1 is in line with detokenization    
    S1 = utils.token_join(S1split, method=tok_method, autocap = autocap)


    S1_score, S2_score = get_CE_scores([S1], S2, tokenizer, model, max_tokens_batch) #, tok_method = tok_method)
    score_S1_og = S1_score[0] # need these for the filter function later
    score_S2_og = S2_score[0]
    og = {'S1_':S1 , 'split_S1_': S1split,
          'CE_S1_': score_S1_og,'CE_S2':score_S2_og,
          'parent_CE':score_S1_og + 1.}
    parent_CE[tuple(S1split)] =    score_S1_og + 1. 
    parent_len[tuple(S1split)] =    len(S1split)


    # length of the original sentence S1
    len_S1 = len(og['split_S1_'])
    
    # S1_list is a nested list, where the ith sublist is a slot for 
    # summary candidates of length i (populated from left to right)
    S1_list = [[]]*(len_S1 + 1) 
    S1_list[len_S1] = [og]#[{'S1_':opt.S1 , 'S2_score':score_S2_og, 'S1_score':score_S1_og} ]
    
    
    # candidate_list_S1_ is a list of lists of candidate sentences (as lists of tokens)
    # this is a way to collect sentences of a given token length before filtering for
    # duplicates and then scoring etc.
    candidat_list_S1_ = [[]]*(len_S1 + 1)
    candidat_list_S1_[len_S1] = [og['split_S1_']]
    

    # define functions for scoring and filtering
    score_beam = lambda v: v['CE_S2']
    filter_fun = lambda v: v['CE_S1_'] <= v['parent_CE']


    ##
    # Do beam search over eliminations
    ##
    for i in range(len_S1, min_words, -1): # populate S1_list with sentences of length 

        
        ##
        # 1 score candidates for this round
        ##
        
        # get token-split candidates for this round
        candidates_split = candidat_list_S1_[i]
        
        # remove duplicates
        as_tuple = [tuple(v) for v in candidates_split]  
        candidates_split = [list(v) for v in list(set(as_tuple))]
        
        # score candidates
        candidates = [utils.token_join(candidate, method = tok_method, autocap = autocap) for candidate in candidates_split] # get this as strings
        
        # get scores for candidates
        S1_scores, S2_scores = get_CE_scores(candidates, S2, tokenizer, model, max_tokens_batch) #, tok_method = tok_method) 

        # this will hold the scoring dicts for the candidates
        candidates_dict = []
        
        ## for each candidate, create score dict and add it to S1_list[i]
        for j, candidate_split in enumerate(candidates_split):
            len_can = len(candidate_split)
            candidate = candidates[j]
            assert(len_can == i) # should be shortened

            # scoring dictionary for S1_ 
            candidate_dict = {'S1_': candidate, 'split_S1_':candidate_split,
                              'CE_S2':S2_scores[j], 'CE_S1_':S1_scores[j],
                              'parent_CE':parent_CE[tuple(candidate_split)],
                              'parent_len':parent_len[tuple(candidate_split)]}

            candidates_dict = candidates_dict + [candidate_dict]
        

        candidates_dict = list(filter(filter_fun, candidates_dict))
        
        ##
        # 2 filter scored candidates if this option is selected
        ##

        S1_list[i] = candidates_dict # store these scores in the overall list
        
        # If we have no candidates after filtering, beam search is done
        if len(candidates_dict) == 0:
            break
        
        ##
        # 3 get the topk options and expand them
        ##
        
        
        # get list of options of this length sorted
        candidates_dict = sorted(candidates_dict, key=score_beam)
        
        topk_list = []
        topk_CE_list = []

        ind = 0
        while (len(topk_list) < k and ind < len(candidates_dict)):
            # if this will not be a duplicate, add it to the beam
            if candidates_dict[ind]['S1_'] not in topk_list: # if the sentence for this expansion is unique, in topk list, add it
                topk_list += [candidates_dict[ind]['split_S1_']] #ignore scores, 
                topk_CE_list += [candidates_dict[ind]['CE_S1_']]
            ind+=1
        

        
        # there should have been no duplicates in expansion
        assert(len(topk_list) == ind)
        # there should also be no duplicates in topk_list
#        assert(len(topk_list) == len(list(set(topk_list)))) # CAN'T HASH LIST
        
        
        expansions_split = []
        
        for k_, topk in enumerate(topk_list):
            expansions_split_iter = utils.get_expansions([topk_list[k_]], rem_words, consecutive = True, tok_method = tok_method, window = window)
            expansions_split += expansions_split_iter
            
            for expansion in expansions_split_iter:     
                as_tup = tuple(expansion)
                parent_len[as_tup] = len(topk)
                if as_tup in parent_CE.keys():

                    if parent_version == 'min':
                        parent_CE[as_tup] = min([parent_CE[as_tup], topk_CE_list[k_]])
                    elif parent_version == 'max':
                        parent_CE[as_tup] = max([parent_CE[as_tup], topk_CE_list[k_]])
                    else: 
                        assert(False)
                else: 
                    parent_CE[as_tup] = topk_CE_list[k_]
        
        ##
        # 4 put expanded options into lists based on length
        ##
        
        for expansion in expansions_split:
            len_exp = len(expansion)
            candidat_list_S1_[len_exp] = candidat_list_S1_[len_exp] + [expansion]
        
  
    S1_list = flatten(S1_list)  

 
    # do final filtering
    S1_list = list(filter(filter_fun, S1_list))
    assert(len(list(S1_list)) > 0)

    # final scoring is score beam?
    scoring_fun = score_beam
    

    S1_out = min(S1_list, key=scoring_fun)

    
    og['score'] = scoring_fun(og)
    S1_out['score'] = scoring_fun(S1_out)
    
    
    return (og,S1_out)
        

    