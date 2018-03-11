#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 8 2018

@author: Felipe Melo

Based on the following tutorial:
youtube.com/watch?v=ElmBrKyMXxs
"""

import numpy as np
import tensorflow as tf

tf.reset_default_graph()

tf.__version__

# Defining Some Constants

start_token = '\t'
end_token = '\n'
batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000
data_path = 'data/fra.txt'

### Data Preprocessing ###

input_seqs = []
target_seqs = []
input_tokens = set()
target_tokens = set()

with open(data_path, 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    
for line in lines[: min(num_samples, len(lines)-1)]:
    input_seq, target_seq = line.split('\t')
    
    target_seq = start_token + target_seq + end_token
    
    input_seqs.append(input_seq)
    target_seqs.append(target_seq)
    
    for token in input_seq:
        if token not in input_tokens:
            input_tokens.add(token)
            
    for token in target_seq:
        if token not in target_tokens:
            target_tokens.add(token)
            
input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

max_encoder_seq_length = max([len(seq) for seq in input_seqs])
max_decoder_seq_length = max([len(seq) for seq in target_seqs])         



