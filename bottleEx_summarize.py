#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:12:54 2019

@author: peterawest
"""

from utils.bottleEx_extractive_funs import elimination_beam_search
import argparse
from tqdm import tqdm
import time
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel



        
    



def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-S1_path', type=str,default = '') # path to S1s (by line)
    parser.add_argument('-S2_path', type=str,default = '') # path to S2s (by line)
    
    
    parser.add_argument('-out_name', type=str, default = '') # name of file to output summaries to
    parser.add_argument('-max_tokens_batch',type=int, default=20000) # max tokens to process per batch (depends on mem)
    parser.add_argument('-start',type=int, default=0) # first example to process (offset)
    parser.add_argument('-log_interval', type = int, default = 50)
    parser.add_argument('-window', type = int, default = 50) # window in which to carry out deletions (for very long S1)
    
    parser.add_argument('-rem_words',type=int, default=2) # max words to remove per step
    parser.add_argument('-beam',type=int, default=1) # number of candidates to keep at each step
    parser.add_argument('-min_words', type=int, default = 1) # minimum number of words to include in a summary
    
    parser.add_argument('-lowercase', action='store_true')
    
    opt = parser.parse_args()
    
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
    
    
    out_str = ''
    count = 0
    start = opt.start
    
    start_time = time.time()


    examples_S1 = []
    examples_S2 = []
    
    with open(opt.S1_path,'r') as f_s1, open(opt.S2_path,'r') as f_s2:
        for line in f_s1.readlines():
            examples_S1 += [line.rstrip()]

        for line in f_s2.readlines():
            examples_S2 += [line.rstrip()]
            
        assert(len(examples_S1) == len(examples_S2))
    examples = list(zip(examples_S1, examples_S2))
        
    
    for example in tqdm(examples, mininterval=2, desc='  - (Generating from examples)   ', leave=False):

        # at log_interval, save intermediate output and output stats
        if count % opt.log_interval == 0:
            with open(opt.out_name, 'w') as f:
                f.write(out_str)
            c_time = time.time() - start_time
            print('count {} || time {} || {} s/it'.format(count, c_time, c_time/(count - start + 0.00001)))
        
        # skip until reaching start index
        if count < start:
            count += 1
            continue
        
        # get S1 and S2
        S1 = example[0]
        S2 = example[1]

        # summarize!
        result = elimination_beam_search(S1 = S1, S2 = S2,
                            k = opt.beam, 
                            rem_words = opt.rem_words, 
                            max_tokens_batch = opt.max_tokens_batch,
                            tok_method = 'moses',
                            autocap = True,
                            window = opt.window,
                            model = model,
                            tokenizer = tokenizer,
                            min_words = opt.min_words)
        
        # process output summary
        out_summ = result[1]['S1_']
        if opt.lowercase:
            out_summ = out_summ.lower()
        out_str += '{}\n'.format(out_summ)

        count += 1

    # save all summaries at the end
    with open(opt.out_name, 'w') as f:
        f.write(out_str)            

if __name__ == '__main__':
    main()
