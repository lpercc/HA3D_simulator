'''
Author: Dylan Li dylan.h.li@outlook.com
Date: 2024-03-17 21:42:00
LastEditors: Dylan Li dylan.h.li@outlook.com
LastEditTime: 2024-03-26 21:15:35
FilePath: /HC3D_simulator/tasks/HC/datasets.py
Description: 

Copyright (c) 2024 by Heng Li, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, check_agent_status
from env import R2RBatch
from model import EncoderLSTM, AttnDecoderLSTM
from agent import Seq2SeqAgent, RandomAgent
import pickle
from eval import Evaluation
import json
import gzip

from transformers import BartTokenizer, BartModel

import sys
module_path = '/home/dylan/projects/motion_hcl/Matterport3DSimulator/build'
if module_path not in sys.path:
    sys.path.append(module_path)
    
    

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'tasks/R2R/snapshots/'
PLOT_DIR = 'tasks/R2R/plots/'
TRAJS_DIR = 'tasks/R2R/trajs'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80

features = IMAGENET_FEATURES
batch_size = 100
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = 'sample' # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
n_iters = 5
model_prefix = 'seq2seq_%s_imagenet' % (feedback_method)



def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_random(train_env, n_iters, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    agent = RandomAgent(train_env, "")
    
    print('Random Agent Begins')
    trajs = []

    for _ in tqdm(range(0, n_iters)):
        
        # traj is a list of dictionaries, each of which is a episode, we have a batch of episodes
        traj = agent.rollout()
        trajs.extend(traj)
    
    return trajs 
        
def train_random_run(iter):
    ''' Train on the training set, and validate on seen and unseen splits. '''

    setup()
    # Create a batch training environment that will also preprocess text
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tok = BartTokenizer.from_pretrained('facebook/bart-base')
    embedding_model = BartModel.from_pretrained('facebook/bart-base')
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok, text_embedding_model=embedding_model, device=device)

    for i in range(iter):
    # Build models and train
        trajs = train_random(train_env, n_iters)
            
        # save trajs as a json 
        with open(TRAJS_DIR + f'/train_trajs_{i}.pkl', 'wb') as f:
            pickle.dump(trajs, f)

if __name__ == '__main__':
    train_random_run(1)