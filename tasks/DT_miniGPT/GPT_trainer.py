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
from env import HABatch
from eval import Evaluation
logger = logging.getLogger(__name__)
import json
import random
import cv2
import torch
from datetime import datetime
HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")
RESULT_DIR = os.path.join(HA3D_SIMULATOR_PATH, 'tasks/DT_miniGPT/results/')
TENSORBOARD_DIR = os.path.join(HA3D_SIMULATOR_PATH, "tasks/DT_miniGPT/tensorboard_logs/")
class Trainer: 
    
    def __init__(self, model, train_dataset, val_seen_dataset, val_unseen_dataset, args, model_dir):
        self.model = model
        self.train_dataset = train_dataset
        self.val_seen_dataset = val_seen_dataset
        self.val_unseen_dataset = val_unseen_dataset
        self.args = args
        self.model_dir = model_dir
        # take over whatever gpus are on the system
        self.device = f'cuda:{self.args.cuda}' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device) #TODO: Add dataparallel in server 
        self.scale_writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, f'{args.experiment_id}_{self.args.fusion_type}_{self.args.feedback_method}_{self.args.reward_strategy}'))
        self.features = os.path.join(HA3D_SIMULATOR_PATH, f'img_features/{self.args.features}.tsv')
        self.tok = BartTokenizer.from_pretrained("facebook/bart-base")
        self.embedding_model = BartModel.from_pretrained("facebook/bart-base")
        self.hparams = {
            'batch_size' : self.args.batch_size,
            'max_episode_len' : self.args.max_episode_len,
            'reward_strategy' : int(self.args.reward_strategy.split('_')[-1]),
            'fusion_type' : int(['bert', 'simple', 'attention'].index(self.args.fusion_type)),
            'target_rtg' : self.args.target_rtg,
        } 
        print(f'Hyparams:{self.hparams}')
    def save_checkpoint(self, path):
        # DataParallel wrappers keep raw model object in .module attribute
        #raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", path)
        torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path):
        self.saved_epoch = int(path.split('/')[-1].split('.')[0].split('_')[-1])
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)
        
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
                    pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            
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
            self.scale_writer.add_scalar('Loss/train', train_loss, epoch)
            self.scale_writer.add_scalar('Loss/val_seen_loss', val_seen_loss, epoch)
            self.scale_writer.add_scalar('Loss/val_unseen_loss', val_unseen_loss, epoch)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record_file = open(os.path.join(self.model_dir, "train_log.txt"), 'a')
            record_file.write(f"{current_time} Epoch: {epoch} Train Loss: {train_loss} val_seen_loss: {val_seen_loss} val_unseen_loss: {val_unseen_loss}\n")
            record_file.close() 

            if epoch % self.args.save_interval == 0:
                save_model_path = os.path.join(self.model_dir, 
                                    f"model_epoch_{epoch}.pth")
                self.save_checkpoint(save_model_path)
                #record_file = open(os.path.join(self.model_dir, "train_log.txt"), 'a')
                #self.saved_epoch = epoch
                # eval_results, eval_results_dict = self.val()
                # record_file.write(f"{eval_results}\n")
                # record_file.close() 

        self.scale_writer.close()

    def val(self):
        """Init a env to evaluate decision transformer"""
        self.val_envs = {split: (HABatch(self.features,                                        
                                batch_size=300 if self.args.batch_size > 300 else self.args.batch_size, 
                                splits=[split],
                                tokenizer=self.tok, 
                                text_embedding_model=self.embedding_model, 
                                device=self.device), Evaluation([split])) for split in ['val_seen', 'val_unseen']}
        eval_results = ''
        eval_results_dict = {}
        for env_name, (env, evaluator) in self.val_envs.items():
            result_file = os.path.join(RESULT_DIR, f'{self.args.experiment_id}_{self.args.model_name}_{self.args.feedback_method}_{self.args.reward_strategy}_{self.saved_epoch}', f"{env_name}_result.json")
            agent = DecisionTransformerAgent(
                env, result_file, self.model
            )
            assert self.args.reward_strategy != ''
            agent.set_reward(self.args.reward_strategy)
            if self.args.mode != 'debug':
                agent.test()
                agent.write_results()
            score_summary, _ = evaluator.score(result_file)
            eval_results += f'{env_name}:'
            eval_results_dict[env_name] = {}
            for metric,val in score_summary.items():
                eval_results += ', %s: %.3f' % (metric, val)
                eval_results_dict[env_name][str(metric)] = float('%.3f' % val)
            eval_results += '\n'
        print(eval_results)
        write_eval_tensorboard(self.scale_writer, eval_results_dict, self.saved_epoch)
        return eval_results, eval_results_dict
        
    def get_trained_model(self):
        return self.model


def write_eval_tensorboard(scale_writer, eval_results_dict, epoch):
    for env_name, score_dict in eval_results_dict.items():
        for metric, val in score_dict.items():
            scale_writer.add_scalar(f'{metric}/{env_name}', val, epoch)

def write_eval_tensorboard_hparams(hparam_writer, hparams, train_loss, val_seen_loss, val_unseen_loss, eval_results_dict):
    metrics = {
        'train_loss' : train_loss,
        'val_seen_loss' : val_seen_loss,
        'val_unseen_loss' : val_unseen_loss,
    }
    for env_name, score_dict in eval_results_dict.items():
        for metric, val in score_dict.items():
            if metric == 'spl':
                continue
            metrics[f"{env_name}/{metric}"] = val
    hparam_writer.add_hparams(hparams, metrics)