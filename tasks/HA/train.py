import os
import sys
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from agent import Seq2SeqAgent
from env import HABatch
from eval import Evaluation
from model import AttnDecoderLSTM, EncoderLSTM
from torch import optim
from param import args
from utils import (
    Tokenizer,
    build_vocab,
    check_agent_status,
    padding_idx,
    read_vocab,
    timeSince,
    write_vocab,
)

HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")

TRAIN_VOCAB = os.path.join(HA3D_SIMULATOR_PATH, f'tasks/HA/data/train_vocab.txt')
TRAINVAL_VOCAB = os.path.join(HA3D_SIMULATOR_PATH, f'tasks/HA/data/trainval_vocab.txt')
RESULT_DIR = os.path.join(HA3D_SIMULATOR_PATH, f'tasks/HA/results/')
SNAPSHOT_DIR = os.path.join(HA3D_SIMULATOR_PATH, f'tasks/HA/snapshots/')
PLOT_DIR = os.path.join(HA3D_SIMULATOR_PATH, f'tasks/HA/plots/')

MAX_INPUT_LENGTH = args.max_input_length

features = os.path.join(HA3D_SIMULATOR_PATH, f'img_features/{args.features}.tsv')
batch_size = args.batch_size
max_episode_len = args.max_episode_len
word_embedding_size = args.word_embedding_size
action_embedding_size = args.action_embedding_size
hidden_size = args.hidden_size
bidirectional = args.bidirection
dropout_ratio = args.dropout_ratio
feedback_method = args.feedback_method # teacher or sample
learning_rate = args.learning_rate
weight_decay = args.weight_decay
n_iters = args.n_iters
model_prefix = args.model_prefix
actionLevel = args.action_level
print(str(args)+"\n")
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
    train_env = HABatch(features, batch_size=batch_size, splits=['train', 'val_seen', 'val_unseen'], tokenizer=tok)

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters)

    # Generate test submission
    test_env = HABatch(features, batch_size=batch_size, splits=['test'], tokenizer=tok)
    agent = Seq2SeqAgent(test_env, "", encoder, decoder, max_episode_len)
    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, 'test', 20000)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()

def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''

    setup()
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = HABatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok, device=device)

    # Creat validation environments
    val_envs = {split: (HABatch(features, batch_size=batch_size, splits=[split],
                tokenizer=tok, device=device), Evaluation([split])) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters, val_envs=val_envs)
        
def valid_teacher():
    setup()
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = HABatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok, device=device)

    # Creat validation environments
    val_envs = {split: (HABatch(features, batch_size=batch_size, splits=[split],
                tokenizer=tok, device=device), Evaluation([split])) for split in ['train', 'val_seen', 'val_unseen']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).cuda()
    torch.set_grad_enabled(False)

    agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len)

    for env_name, (env, evaluator) in val_envs.items():
        agent.env = env
        agent.results_path = '%s%s_%s_iter_%s.json' % (RESULT_DIR, model_prefix, env_name, actionLevel)
        #agent.test(use_dropout=False, feedback='argmax', iters=iters)
        agent.test_teacher(args.action_level)
        agent.write_results()
        #agent.write_results()
        if env_name != '' and (not env_name.startswith('test')):
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

            record_file = open(os.path.join(PLOT_DIR, f'teacher_{args.action_level}_local3_valid_log.txt'), 'a')
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
    #valid_teacher()
    #test_submission()
