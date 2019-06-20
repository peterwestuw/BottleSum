# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
          --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
          --output_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange
from math import ceil

import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                     OpenAIAdam, cached_path, OpenAIGPTLMHeadModel,
                                     GPT2Tokenizer, GPT2LMHeadModel) #, WEIGHTS_NAME, CONFIG_NAME)

from test_generating import generate_from_model
from torch.nn import CrossEntropyLoss

    
def eval_model(model, nbatch_eval,batches_eval,labels_eval,bsz):
    
    model.eval()
    
    eval_loss = 0
    
    
    for i_batch in tqdm(list(range(nbatch_eval)), desc="Evaluating"):
        with torch.no_grad():
            batch = batches_eval[i_batch]#X_eval[:, i_batch*bsz:(1+i_batch)*bsz]
            if batch.numel() == 0:
                break
            
            batch = batch.cuda()
            lm_labels = labels_eval[i_batch].cuda()
            loss_fct = CrossEntropyLoss(reduction = 'none')
            lm_logits,_ = model(batch)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:,1:].contiguous()
            
            shift_labels_mask = (lm_labels[:,1:].contiguous().view(-1) != -1).float()
            
            loss_mat = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
            loss = (loss_mat*shift_labels_mask).view(-1).sum()/shift_labels_mask.sum() # avg over non-masked indices
                



#            loss = model(batch*(batch!=-1).long(), lm_labels = batch)
            eval_loss += loss.item()
    return eval_loss


def to_json_file(self, json_file_path):
    """ Save this instance to a json file."""
    with open(json_file_path, "w", encoding='utf-8') as writer:
        writer.write(self.to_json_string())

WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME = 'config.json'

ROCSTORIES_URL = "https://s3.amazonaws.com/datasets.huggingface.co/ROCStories.tar.gz"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)




def load_dataset_0(file_name, tokenizer, bptt = 35, bsz =35, label_s1 = True, delim = ' TL;DR: ', end_tok = '^'):
        
    enc = tokenizer.encode
#    enc = lambda a: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(a))
    
    with open(file_name, 'r') as file:
        lines = file.readlines()
    
    batches = []    
    X = []
    
    batch = []
    tokens_batch = 0
    labels = []
#    for line in lines:
    for line in tqdm(list(lines), mininterval=2, desc='  - (Loading data)   ', leave=False):

        seq = line.rstrip()
        
        try:
            enc_seq = enc(seq)
            if len(enc_seq) < 200:
                batch += [seq]
                tokens_batch += len(enc_seq)
        except:
            pass

        # if this batch is full, process it
        if len(batch) == bsz:
            
            if label_s1: # train on the original sentence
                batch = [enc(seq) for seq in batch]
                
                max_len = max([len(seq) for seq in batch])
                batch = [torch.tensor(seq + [-1]*(max_len - len(seq))).view(1,max_len) for seq in batch] #pad with -1
                X = torch.cat(batch, dim=0)
                X_batch = X*(X!=-1).long()
                batches += [X_batch]

                labels += [X]
            else:
#                delim = 'TL;DR: '
                
                batch_s1 = []
                batch_s2 = []
                batch_s = []
                
                
                for s in batch:
                    ind = s.find(delim)
                    s1 = s[:ind + len(delim)]
                    s2 = s[ind + len(delim):]
                    batch_s1 += [enc(s1)]
                    batch_s2 += [enc(s2)]
                    batch_s += [enc(s1) + enc(s2)]
                    
                max_len = max([len(s) for s in batch_s])
                
                batch_label = []
                batch_inp = []
                for i, _ in enumerate(batch):
                    s = batch_s[i]
                    s1 = batch_s1[i]
                    s2 = batch_s2[i]
                    
                    batch_inp += [torch.tensor(s + [0]*(max_len - len(s))).view(1,max_len)]
                    batch_label += [torch.tensor([-1]*len(s1) + s2 + [-1]*(max_len - len(s))).view(1,max_len)]
                
                
                batches += [torch.cat(batch_inp, dim=0)]
                labels += [torch.cat(batch_label, dim=0)]
                

            batch = []
            tokens_batch = 0

    nbatches = len(batches)
    
    return batches, labels, nbatches

def load_dataset(file_source, file_target, tokenizer, bptt = 35, bsz =35, label_s1 = True, delim = ' TL;DR: ', end_tok = 'Δ'):
        
    
    
    enc = tokenizer.encode
#    enc = lambda a: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(a))
    

        
    with open(file_source, 'r') as file:
        lines_source = file.readlines()
    with open(file_target, 'r') as file:
        lines_target = file.readlines()
        
    assert(len(lines_source) == len(lines_target))
    
    batches = []    
    X = []
    
    batch = []
    tokens_batch = 0
    labels = []
#    for line in lines:
    for i in tqdm(range(len(lines_source)), mininterval=2, desc='  - (Loading data)   ', leave=False):

        s_source = lines_source[i].rstrip()
        
        s_target = lines_target[i].rstrip()
        
        # delim should not appear in the data
        assert(not (end_tok in s_source)) 
        assert(not (end_tok in s_target))
        
        seq = s_source + ' {} '.format(delim) + s_target + end_tok
        #seq =   line.rstrip()
        
        try:
            enc_seq = enc(seq)
            if len(enc_seq) < 200:
                batch += [seq]
                tokens_batch += len(enc_seq)
        except:
            print('skipping line')
            continue
            pass

        # if this batch is full, process it
        if len(batch) == bsz:
            
            if label_s1: # train on the original sentence
                batch = [enc(seq) for seq in batch]
                
                max_len = max([len(seq) for seq in batch])
                batch = [torch.tensor(seq + [-1]*(max_len - len(seq))).view(1,max_len) for seq in batch] #pad with -1
                X = torch.cat(batch, dim=0)
                X_batch = X*(X!=-1).long()
                batches += [X_batch]

                labels += [X]
            else:
#                delim = 'TL;DR: '
                
                batch_s1 = []
                batch_s2 = []
                batch_s = []
                
                
                for s in batch:
                    ind = s.find(delim)
                    s1 = s[:ind + len(delim)]
                    s2 = s[ind + len(delim):]
                    batch_s1 += [enc(s1)]
                    batch_s2 += [enc(s2)]
                    batch_s += [enc(s1) + enc(s2)]
                    
                max_len = max([len(s) for s in batch_s])
                
                batch_label = []
                batch_inp = []
                for i, _ in enumerate(batch):
                    s = batch_s[i]
                    s1 = batch_s1[i]
                    s2 = batch_s2[i]
                    
                    batch_inp += [torch.tensor(s + [0]*(max_len - len(s))).view(1,max_len)]
                    batch_label += [torch.tensor([-1]*len(s1) + s2 + [-1]*(max_len - len(s))).view(1,max_len)]
                
                
                batches += [torch.cat(batch_inp, dim=0)]
                labels += [torch.cat(batch_label, dim=0)]
                

            batch = []
            tokens_batch = 0

    nbatches = len(batches)
    
    return batches, labels, nbatches




def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
        lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch,), dtype=np.int64)
        for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
            with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
            with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
            input_ids[i, 0, :len(with_cont1)] = with_cont1
            input_ids[i, 1, :len(with_cont2)] = with_cont2
            mc_token_ids[i, 0] = len(with_cont1) - 1
            mc_token_ids[i, 1] = len(with_cont2) - 1
            lm_labels[i, 0, :len(with_cont1)-1] = with_cont1[1:]
            lm_labels[i, 1, :len(with_cont2)-1] = with_cont2[1:]
            mc_labels[i] = mc_label
        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default='tuned_gpt2', type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    
    parser.add_argument('--source_eval', type=str, default='')
    parser.add_argument('--target_eval', type=str, default='')
    parser.add_argument('--source_train', type=str, default='')
    parser.add_argument('--target_train', type=str, default='')
    
    
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--effective_batch_size',type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--bsz', type=int, default = 20)
    parser.add_argument('--bptt', type=int, default = 40)

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
#    print(args)

    model_type = 'gpt2'


    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(type='cuda')
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

#    if not args.do_train and not args.do_eval:
#        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')

    model.to(device)


    #file_train = args.train_dataset #'cnn_train.txt'
    #file_eval =  args.eval_dataset #'cnn_valid.txt'
    bptt = args.bptt
    bsz = args.bsz
    

#    X_eval, nbatch_eval = load_dataset(file_eval, tokenizer, bptt, bsz)
#    X_train, nbatch_train =  load_dataset(file_train, tokenizer, bptt, bsz)
    
    batches_eval, labels_eval, nbatch_eval = load_dataset(args.source_eval, args.target_eval, tokenizer, bptt, bsz)
    batches_train, labels_train, nbatch_train =  load_dataset(args.source_train, args.target_train, tokenizer, bptt, bsz)
    
    

    # Prepare optimizer
#    param_optimizer = list(model.parameters())
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    print('here 3')
#    num_train_optimization_steps = len(train_data) * args.num_train_epochs // args.train_batch_size
    num_train_optimization_steps = nbatch_train * args.num_train_epochs
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           max_grad_norm=args.max_grad_norm,
                           weight_decay=args.weight_decay,
                           t_total=num_train_optimization_steps)

    eval_loss_min = None
    print('here 4')
    model.to(device)

    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
    model.train()
    for epoch_i in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_steps = 0
        
        for i_batch in tqdm(list(range(nbatch_train)), desc='Evaluating epoch {}'.format(epoch_i)):
            batch = batches_train[i_batch]#X_train[:, i_batch*bsz:(1+i_batch)*bsz].permute(1,0)
            
            batch = batch.cuda()
            lm_labels = labels_train[i_batch].cuda()
            if batch.numel() == 0:
                break
            
            #loss = model(batch, lm_labels = labels_train[i_batch].cuda())
                            # TRY DOING IT MANUALLY
            loss_fct = CrossEntropyLoss(reduction = 'none')
            lm_logits,_ = model(batch)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:,1:].contiguous()
            
            shift_labels_mask = (lm_labels[:,1:].contiguous().view(-1) != -1).float()
            
            loss_mat = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
            loss = (loss_mat*shift_labels_mask).view(-1).sum()/shift_labels_mask.sum() # avg over non-masked indices
            
            loss.backward()
            
            # only step the model if you've gone through 'effective_batch_size' examples
            if (i_batch*args.train_batch_size) % args.effective_batch_size == 0 and i_batch != 0:
                optimizer.step()
                optimizer.zero_grad()
                
            tr_loss += loss.item()
            

            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            nb_tr_steps += 1
         
            
            
            ###
            # Evaluations
            ###
            
            
            if i_batch % 1000 == 0: # get eval score
                eval_loss = eval_model(model, nbatch_eval,batches_eval,labels_eval, bsz)
                
                # if eval_loss improves, save model
                if eval_loss_min is None or eval_loss < eval_loss_min:
                    eval_loss_min = eval_loss
                    
                    # save model if eval loss is lower
                    model_to_save = model
                    # If we save using the predefined names, we can load using `from_pretrained`
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        
                    torch.save(model_to_save.state_dict(), output_model_file)
                    to_json_file(model_to_save.config,output_config_file)
                
                print('eval_loss {}',format(eval_loss))
                model.train()
                
            if i_batch % 200 == 0: # try generating from model 
                print("Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0]))

                model.eval()
                if model_type == 'gpt':
                    encode = lambda a: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(a))
                    decode = tokenizer.decode
                elif model_type == 'gpt2':
                    encode = tokenizer.encode
                    decode = tokenizer.decode
                
                generate_from_model(encode, decode, model = model,model_type = model_type)
                model.train()
            


if __name__ == '__main__':
    main()
