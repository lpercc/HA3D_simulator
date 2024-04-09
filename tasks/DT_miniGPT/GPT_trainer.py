'''
Author: Dylan Li dylan.h.li@outlook.com
Date: 2024-03-26 15:10:27
LastEditors: Dylan Li dylan.h.li@outlook.com
LastEditTime: 2024-03-30 09:59:08
FilePath: /motion_hcl/Matterport3DSimulator/tasks/R2R/DT/GPT_trainer.py
Description: 

Copyright (c) 2024 by Heng Li, All Rights Reserved. 
'''
import math
import logging

from tqdm import tqdm
import numpy as np
import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BartModel, BartTokenizer
from agent import DecisionTransformerAgent
from env import HCBatch
from eval import Evaluation
logger = logging.getLogger(__name__)
import json
import random
import cv2
import torch
from datetime import datetime
HC3D_SIMULATOR_PATH = os.environ.get("HC3D_SIMULATOR_PATH")
RESULT_DIR = os.path.join(HC3D_SIMULATOR_PATH, 'tasks/DT_miniGPT/results/')

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    cuda = 0
    log_path = ''
    features = ''
    reward_strategy = ''


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer: 
    
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = f'cuda:{config.cuda}' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.model = self.model.to(self.device) #TODO: Add dataparallel in server 
        self.writer = SummaryWriter(log_dir=config.log_path)

        tok = BartTokenizer.from_pretrained("facebook/bart-base")
        embedding_model = BartModel.from_pretrained("facebook/bart-base")
        self.val_envs = {split: (HCBatch(self.config.features, batch_size=self.config.batch_size, splits=[split],
            tokenizer=tok, text_embedding_model=embedding_model, device=self.device), Evaluation([split])) for split in ['val_seen', 'val_unseen']}
            
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)
        
        
    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_one_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, _, r, t) in pbar: # states, actions, targets, rtgs, timesteps

                # place data on the correct device
                x = x.to(self.device) 
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss
            else: 
                return losses
        
        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            losses = run_one_epoch('train',)
            train_loss = np.mean(losses)
            print("Train Loss: ", train_loss)
            test_loss = run_one_epoch('test')
            print("Test Loss: ", test_loss)
            self.writer.add_scalar('Loss/train', train_loss, epoch+1)
            self.writer.add_scalar('Loss/test', test_loss, epoch+1)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record_file = open(os.path.join(config.log_path, "train_log.txt"), 'a')
            record_file.write(f"{current_time} Epoch: {epoch+1} Train Loss: {train_loss} Test Loss: {test_loss}\n")
            record_file.close() 
        
        self.trained_model = model
        self.writer.close()
        record_file = open(os.path.join(config.log_path, "train_log.txt"), 'a')
        log_info = self.val(self.trained_model)
        record_file.write(f"{log_info}\n")
        record_file.close() 

    def val(self, model):
        """Init a env to evaluate decision transformer"""
        log_info = ''
        for env_name, (env, evaluator) in self.val_envs.items():
            result_file = os.path.join(RESULT_DIR, f"{env_name}_result.json")
            agent = DecisionTransformerAgent(
                env, result_file, model
            )
            assert self.config.reward_strategy != ''
            agent.set_reward(self.config.reward_strategy)
            agent.test()
            agent.write_results()
            
            score_summary, _ = evaluator.score(result_file)
            log_info += f'{env_name}:'
            for metric,val in score_summary.items():
                log_info += ', %s: %.3f' % (metric, val)

        return log_info
        
    def get_trained_model(self):
        return self.trained_model