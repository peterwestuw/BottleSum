#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:30:40 2019

@author: peterawest
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:54:48 2019

@author: peterawest
"""

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from nltk.tokenize import word_tokenize

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer

def trim_text(text, end_tok, len_s_in):
    '''
    Trim the end of generated text at the first instance of the 
    end token, but also make sure the generation is no longer than the
    input sentnece (len_s_in)
    '''
    
    trimmed_ind = text.find(end_tok)

    
    if trimmed_ind == -1:
        trimmed_ind = len_s_in # default
        text = text[:trimmed_ind] # make sure it's no longer than s_in first
        
        # if it ends with the beginning of end_tok, trim
        for j in range(1, len(end_tok)):
            if text.endswith(end_tok[:j]):
                trimmed_ind = len(text) - j

    trimmed_text = text[:trimmed_ind]
    
    return trimmed_text

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def top_p_logits(logits,p):
    """
    Masks everything but the logits that cover the top-p probability mass
    
    !!! plan: do sm, sort p's by descending size, do cumsum, get index 
    of where we cross p, use that index to get batch mins
    """
    
    # FIGURE OUT DIMENSIONS OF THIS
    sm = torch.nn.Softmax(dim=0)
    probs = sm(logits.view(-1))
    
    A_sort, sort_inds = probs.sort(descending = True)
    
    cs = A_sort.cumsum(dim=0)
    mask_inds = sort_inds[cs > p]
    
    probs[mask_inds] = -1e10
    return probs.view(1,-1)



def sample_sequence_og(model, length,args, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


def sample_sequence_nucleus(model, length,args, start_token=None, batch_size=None, context=None, temperature=1, top_p=1.0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_p_logits(logits, p=top_p)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


def sample_sequence_beam(model, length, args, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True, beam_size = 5, tokenizer= None, max_len = -1, min_len = -1):
    '''
    Use beam search to sample a sequence conditioned on the given context
    '''
    
    
    
    
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)

    past = None
    
    
    # if specified, limit max generation length to input sentence length
    if args.max_len_inp:
        length = min([length, context.numel() + 1]) # +1 for endchar
    
#    full_beam = []
    
    candidates = [{'prev':torch.tensor(context), 'output':torch.tensor(context),'past':None,'ended':False, 'score':0}]
    
    done_list = []
    
    with torch.no_grad():
        for i in trange(length):
             # the beam for the ith place
            
            candidates_sorted = sorted(candidates, key=lambda v: v['score'])
            k_best = candidates_sorted[:beam_size] # get the best ones
            
            
            candidates = []
            for cand in k_best:
                
                past = None#cand['past']
                prev = torch.tensor(cand['output'])#cand['prev']
                output_0 = torch.tensor(cand['output'])
                

                logits, past = model(prev, past=past)
                logits = logits[:, -1, :] / temperature
                logits = top_k_logits(logits, k=top_k)
                log_probs = F.softmax(logits, dim=-1)
                
                vals, prev = torch.topk(log_probs, k=beam_size, dim=-1)
                
                vals = vals.view(beam_size)
                
                for j in range(prev.numel()): #for each candidate expansion

                    output = torch.cat((torch.tensor(output_0), torch.tensor(prev[:,j].view(1,1))), dim=1)
                    
                    
                    str_out = tokenizer.decode(output[0,:].tolist())
                    score = cand['score'] - vals[j].item()
                    
                    done = args.end_tok in str_out #'<|endoftext|>' in str_out
                    
                    cand_out = {'prev':prev, 'output':output.data,'past':past, 'score':score,'str_out':str_out}
                    
                    # if contains end token or reached end length, dump into done
                    if done or i == length - 1: 
                        # put in the done pile
                        done_list += [cand_out]
                        
                        
                    else: 
                        # put in the beam
                        candidates += [cand_out]
        
        if max_len != -1: # if we have specified a max length
            tmp_done_list = []
            
            for d in done_list:
                
                # remove '<|endoftext|>' if applicable
                str_out = tokenizer.decode(d['output'][0,:].tolist()) 
                trimmed_ind = str_out.find(args.end_tok)#('<|endoftext|>')
                if trimmed_ind == -1:
                    trimmed_ind = len(str_out)
                str_out = str_out[:trimmed_ind]
                tok_len = len(tokenizer.encode(str_out))
                
                if tok_len <= max_len + 1:
                    print('encoded {}'.format(tokenizer.encode(str_out)))
                    tmp_done_list += [d]
            done_list = tmp_done_list
                
        if min_len != -1:
            # trim all of the out strings so min_len is meaningful
            done_list_tmp = []
            for d in done_list:
                str_out = tokenizer.decode(d['output'][0,:].tolist()) 
                str_out = trim_text(str_out, args.end_tok, 10000) # remove end_token if you need to

                if len(tokenizer.encode(str_out)) >= context.numel() + min_len:
                    done_list_tmp += [d]
            done_list = done_list_tmp
                   
        output = max(done_list, key = lambda v: v['score'])['output']
                    
    return output


def sample_sequence(model, length,args, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True, sample_v ='og', tokenizer = None, beam_size = 5, max_len = -1, min_len = -1):
    
    
    if sample_v == 'og': # sampling as in the original script
        return sample_sequence_og(model, length,args, start_token=start_token, batch_size=batch_size, context=context, temperature=temperature, top_k=top_k, device=device, sample=sample)
    if sample_v == 'top_p':
        return sample_sequence_nucleus(model, length,args, start_token=start_token, batch_size=batch_size, context=context, temperature=temperature, top_p=args.top_p, device=device, sample=sample)
    elif sample_v == 'beam': 
        return sample_sequence_beam(model, length, args, start_token, batch_size, context, temperature, top_k, device, sample, beam_size, tokenizer, max_len = max_len,min_len = min_len)
    else: 
        assert(False)




def generate_from_model(args):

    
    
    with open(args.path_input,'r') as f:
        in_samples = [line.rstrip() for line in f.readlines()]
    
    
    
    
    
    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#    enc = GPT2Tokenizer.from_pretrained('gpt2')
#    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encode = tokenizer.encode
    decode = tokenizer.decode


    

    
    
    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    context_tokens = []
    

    generated = 0
    out_str = ''
    
    for s_in in in_samples:
        
        
        
        in_text= s_in + args.delim_tok 
        context_tokens = encode(in_text)
        
        args.max_len = -1
        if args.len_ratio != -1:
            # set it to a specific value
            args.length = int(len(context_tokens) *args.len_ratio) + 1

        
        print('input: {}\n'.format(in_text))
        out = sample_sequence(
            model=model, length=args.length, args = args,
            context=context_tokens,
            start_token=None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device, sample_v= args.sample_v,
            tokenizer = tokenizer,
            beam_size = args.beam_size,
            max_len = args.max_len,
            min_len = args.min_len
        )
            
            
        out = out[:, len(context_tokens):].tolist()
        if len(out[0]) < args.min_len:
            print(s_in)
            print(out)
            print(len(out[0]))
            assert(False)
        
        for i in range(args.batch_size):
            generated += 1
            text = decode(out[i])
            print("=" * 40 + '\n\n' + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)
            
            trimmed_text = trim_text(text, args.end_tok, len(s_in)) # trim text (see def above)
            
            
            
            print('trimmed: {}'.format(trimmed_text))
            out_str += trimmed_text + '\n'
    print("=" * 80)
    
    with open(args.out_path, 'w') as f:
        f.write(out_str)
    
    
def main():
    parser = argparse.ArgumentParser()
    
    args = lambda a: None
    
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--model_name_or_path',type=str, default='tuned_gpt2')
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--nsamples',type=int, default=1)
    parser.add_argument('--length',type=int, default=60)
    
    parser.add_argument('--max_len_inp', action='store_true') # whether to constrain to be no longer than input
    
    parser.add_argument('--temperature',type=float, default=1.0)
    parser.add_argument('--top_k',type=int, default=0)
    parser.add_argument('--unconditional',action='store_true')
    parser.add_argument('--path_input',type=str, default='')
    parser.add_argument('--len_ratio',type=float, default=-1.) # ratio of original sentence as limit

    parser.add_argument('--end_tok',type=str, default='^')
    parser.add_argument('--delim_tok',type=str, default=' TL;DR: ')
    parser.add_argument('--out_path',type=str, default='out.txt')
    
    
    parser.add_argument('--sample_v',type=str, default='beam')
    parser.add_argument('--beam_size',type=int, default=1)
    parser.add_argument('--min_len',type=int, default=-1)
    
    parser.add_argument('--top_p',type=float, default=0.9)
    

    args = parser.parse_args()
    generate_from_model(args)
    
if __name__ == '__main__':
    main()
