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

class Trainer: 
    
    def __init__(self, model, train_dataset, val_seen_dataset, val_unseen_dataset, args, log_path):
        self.model = model
        self.train_dataset = train_dataset
        self.val_seen_dataset = val_seen_dataset
        self.val_unseen_dataset = val_unseen_dataset
        self.args = args
        self.log_path = log_path
        # take over whatever gpus are on the system
        self.device = f'cuda:{self.args.cuda}' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device) #TODO: Add dataparallel in server 
        self.writer = SummaryWriter()
        self.features = os.path.join(HC3D_SIMULATOR_PATH, f'img_features/{self.args.features}.tsv')
        self.tok = BartTokenizer.from_pretrained("facebook/bart-base")
        self.embedding_model = BartModel.from_pretrained("facebook/bart-base")
            
    def save_checkpoint(self, path):
        # DataParallel wrappers keep raw model object in .module attribute
        #raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", path)
        torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path):
        self.saved_epoch = int(path.split('/')[-1].split('.')[0].split('_')[-1])
        self.model.load_state_dict(torch.load(path))
        
    def train(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = raw_model.configure_optimizers(self.args)

        def run_one_epoch(split):
            if split == 'train':
                is_train = True
                loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers)
            elif split == 'val_seen':
                is_train = False
                loader = DataLoader(self.val_seen_dataset, shuffle=True, pin_memory=True,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers)
            elif split == 'val_unseen':
                is_train = False
                loader = DataLoader(self.val_unseen_dataset, shuffle=True, pin_memory=True,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers)
            self.model.train(is_train)
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
                    logits, loss = self.model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    self.model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if self.args.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < self.args.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, self.args.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - self.args.warmup_tokens) / float(max(1, self.args.final_tokens - self.args.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = self.args.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = self.args.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            
            if not is_train:
                val_loss = float(np.mean(losses))
                logger.info(f"{split} loss: {val_loss}")
                return val_loss
            else: 
                return losses
        
        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(1, self.args.epochs+1):

            losses = run_one_epoch('train',)
            train_loss = np.mean(losses)
            print("Train Loss: ", train_loss)
            val_seen_loss = run_one_epoch('val_seen')
            val_unseen_loss = run_one_epoch('val_unseen')
            print("Val_seen Loss: ", val_seen_loss)
            print("Val_unseen Loss: ", val_unseen_loss)
            self.writer.add_scalar('Loss/train', train_loss, epoch+1)
            self.writer.add_scalar('Loss/val_seen_loss', val_seen_loss, epoch+1)
            self.writer.add_scalar('Loss/val_unseen_loss', val_unseen_loss, epoch+1)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record_file = open(os.path.join(self.log_path, "train_log.txt"), 'a')
            record_file.write(f"{current_time} Epoch: {epoch} Train Loss: {train_loss} val_seen_loss: {val_seen_loss} val_unseen_loss: {val_unseen_loss}\n")
            record_file.close() 

            if epoch % self.args.save_interval == 0:
                save_model_path = os.path.join(self.log_path, 
                                    f"mdoel_epoch_{epoch}.pth")
                self.save_checkpoint(save_model_path)
                record_file = open(os.path.join(self.log_path, "train_log.txt"), 'a')
                log_info = self.val()
                record_file.write(f"{log_info}\n")
                record_file.close() 

        self.writer.close()

    def val(self):
        """Init a env to evaluate decision transformer"""
        self.val_envs = {split: (HCBatch(self.features,                                        
                                batch_size=100 if self.args.batch_size > 100 else self.args.batch_size, 
                                splits=[split],
                                tokenizer=self.tok, 
                                text_embedding_model=self.embedding_model, 
                                device=self.device), Evaluation([split])) for split in ['val_seen', 'val_unseen']}
        log_info = ''
        for env_name, (env, evaluator) in self.val_envs.items():
            result_file = os.path.join(RESULT_DIR, f'{self.args.model_name}_{self.args.feedback_method}_{self.args.reward_strategy}_{self.saved_epoch}', f"{env_name}_result.json")
            agent = DecisionTransformerAgent(
                env, result_file, self.model
            )
            assert self.args.reward_strategy != ''
            agent.set_reward(self.args.reward_strategy)
            agent.test()
            agent.write_results()
            
            score_summary, _ = evaluator.score(result_file)
            log_info += f'\n{env_name}:'
            for metric,val in score_summary.items():
                log_info += ', %s: %.3f' % (metric, val)
        print(log_info)
        return log_info
        
    def get_trained_model(self):
        return self.model