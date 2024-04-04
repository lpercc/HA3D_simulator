
import gzip
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import RandomAgent, Seq2SeqAgent, DecisionTransformerAgent
from env import HCBatch
from eval import Evaluation
from model import AttnDecoderLSTM, EncoderLSTM
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from transformers import BartModel, BartTokenizer
from utils import (
    Tokenizer,
    build_vocab,
    check_agent_status,
    padding_idx,
    read_vocab,
    timeSince,
    write_vocab,
)

from DT.minGPT import GPT, GPT1Config, GPTConfig
from dataclasses import dataclass
from DT.utils import seed_everything

HC3D_SIMULATOR_PATH = os.environ.get("HC3D_SIMULATOR_PATH")

TRAIN_VOCAB = os.path.join(HC3D_SIMULATOR_PATH, 'tasks/HC/data/train_vocab.txt')
TRAINVAL_VOCAB = os.path.join(HC3D_SIMULATOR_PATH, 'tasks/HC/data/trainval_vocab.txt')
RESULT_DIR = os.path.join(HC3D_SIMULATOR_PATH, 'tasks/HC/results/')
SNAPSHOT_DIR = os.path.join(HC3D_SIMULATOR_PATH, 'tasks/HC/snapshots/')
PLOT_DIR = os.path.join(HC3D_SIMULATOR_PATH, 'tasks/HC/plots/')

IMAGENET_FEATURES = os.path.join(HC3D_SIMULATOR_PATH, 'img_features/ResNet-152-imagenet_80_16_mean.tsv')
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
n_iters = 5000 if feedback_method == 'teacher' else 20000
model_prefix = 'seq2seq_%s_imagenet' % (feedback_method)


def train(train_env, encoder, decoder, n_iters, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len)
    print('Training with %s feedback' % feedback_method)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_log = defaultdict(list)
    start = time.time()

    for idx in range(0, n_iters, log_every):

        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback_method)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric,val in score_summary.items():
                data_log['%s %s' % (env_name,metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)
            record_file = open(os.path.join(PLOT_DIR, f'{model_prefix}_log.txt'), 'a')
            record_file.write(loss_str + '\n')
            record_file.close()
        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str))

        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR, model_prefix)
        df.to_csv(df_path)

        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        agent.save(enc_path, dec_path)


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def test_submission():
    ''' Train on combined training and validation sets, and generate test submission. '''

    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = HCBatch(features, batch_size=batch_size, splits=['train', 'val_seen', 'val_unseen'], tokenizer=tok)

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters)

    # Generate test submission
    test_env = HCBatch(features, batch_size=batch_size, splits=['test'], tokenizer=tok)
    agent = Seq2SeqAgent(test_env, "", encoder, decoder, max_episode_len)
    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, 'test', 20000)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''

    setup()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = HCBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok, device=device)

    # Creat validation environments
    val_envs = {split: (HCBatch(features, batch_size=batch_size, splits=[split],
                tokenizer=tok, device=device), Evaluation([split])) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters, val_envs=val_envs)
    #valid_teacher(train_env, encoder, decoder, val_envs=val_envs)
    
def eval_DT():

    ''' Init a env to evaluate decision transformer'''
    setup() 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tok = BartTokenizer.from_pretrained('facebook/bart-base')
    embedding_model = BartModel.from_pretrained('facebook/bart-base')
    #train_env = HCBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok, text_embedding_model=embedding_model, device=device)
    
    # Create Validation Environments 
    val_env = HCBatch(features, batch_size=batch_size, splits=['val_seen'], tokenizer=tok, text_embedding_model=embedding_model, device=device)

    # load models 
    mconf = GPT1Config(6, 5 * 3, model_type = 'reward_conditioned', max_timestep=29)
    model = GPT.load('/home/qid/minghanli/HC3D_simulator/tasks/HC/DT/models/GPT_model.pth', mconf)
    
    val_seen_agent = DecisionTransformerAgent(val_env, '/home/qid/minghanli/HC3D_simulator/tasks/HC/results', model)
    
    traj = val_seen_agent.rollout()
        
def valid_teacher(train_env, encoder, decoder, val_envs={}):
    torch.set_grad_enabled(False)

    agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len)


    for env_name, (env, evaluator) in val_envs.items():
        agent.env = env
        agent.results_path = '%s%s_%s_iter_%s.json' % (RESULT_DIR, model_prefix, env_name, env_name)
        #agent.test(use_dropout=False, feedback='argmax', iters=iters)
        agent.test_teacher()
        agent.write_results()
        #agent.write_results()
        if env_name != '' and (not env_name.startswith('test')):
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

            record_file = open(os.path.join(PLOT_DIR, 'teacher_Hn_valid_log.txt'), 'a')
            record_file.write(loss_str + '\n')
            record_file.close()


if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(RESULT_DIR)):
        os.makedirs(os.path.dirname(RESULT_DIR))
    if not os.path.exists(os.path.dirname(SNAPSHOT_DIR)):
        os.makedirs(os.path.dirname(SNAPSHOT_DIR))
    if not os.path.exists(os.path.dirname(PLOT_DIR)):
        os.makedirs(os.path.dirname(PLOT_DIR))
    #eval_DT()
    train_val()
    #test_submission()
