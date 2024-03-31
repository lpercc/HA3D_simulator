'''
Author: Dylan Li dylan.h.li@outlook.com
Date: 2024-03-26 15:31:50
LastEditors: Dylan Li dylan.h.li@outlook.com
LastEditTime: 2024-03-30 22:34:04
FilePath: /motion_hcl/Matterport3DSimulator/tasks/R2R/DT/train_GPT.py
Description: 

Copyright (c) 2024 by Heng Li, All Rights Reserved. 
'''



import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from minGPT import GPT, GPT1Config, GPTConfig
from GPT_trainer import Trainer, TrainerConfig 
from utils import sample, seed_everything
from collections import deque
import random
import torch
import os 
import pickle
import gzip
#import blosc
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=32)
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
args = parser.parse_args()

seed_everything(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, targets, done_idxs, rtgs, timesteps):        
        """ This is a dataset class for the State-Action-Reward dataset.
        How do we model a GPT like model here? For data, we concanate all text into one list than split it by context windows. 

        Args:
            data (np.arryas): should be a list of state_features 
            block_size (arryas): The is real transformer context length . block_size // 3 will be the input context length for actions, rewards and so on
            actions (arryas]): actions should be a list of integers.
            done_idxs (list[int]): A list of indices where the episode ends. 
            rtgs (arryas): A list of return to go rewards.
            timesteps (arryas): A list of timesteps
            
        return: 
            states
            actions
            rtgs
            timesteps 
        """
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.targets = targets
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        
        assert self.block_size // 3 < np.min(done_idxs), "Error: block_size is too large, should be smaller than the minimum done index, which means that the block_size is too large for the shortest episode." # TODO: shall we set a limit like this in GPT model.
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3 # Set 3 here, because we have 3 inputs: states, actions, rtgs
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, feature_size)
        targets = torch.tensor(self.targets[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1) # (block_size, 1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1) # (block_size, 1)
        
        return states, actions, targets, rtgs, timesteps
    
def load_data(data_dir): 
    trajs = []
    for i, data in enumerate(os.listdir(data_dir)):
        with open(data_dir + f'/train_trajs_{0}.pkl', 'rb') as f: #DONE: change to support pkl 
            traj = pickle.load(f) # 
            trajs.extend(traj)
        break #NOTE: Test with one file 
    return trajs

def create_dataset(trajs):
    # This is a function to read all trajs into one big dataset. 
    states = [] 
    actions = []
    rewards = [] 
    actions = [] 
    targets =  []
    done_idxs = []
    for t in trajs: 
        states.extend(t['state_features'])
        actions.extend(t['actions'])
        rewards.extend(t['final_reward'])
        targets.extend(t['teacher'])
        done_idxs.append(len(t['actions']) - 1) # -1 because the index starts from 0
        
    # Convert to numpy arrays
    states  = np.array(states)
    targets = np.array(targets)
    rewards = np.array(rewards)
    actions = np.array(actions)
    
    assert np.sum(done_idxs) == len(actions) - len(done_idxs), "Error: sum of done_idxs is not equal to length of actions"
    
    
    # -- create return to go reward datasets TODO: return 是错位的
    rtgs = np.zeros(len(rewards))
    start_index = 0
    for done_idx in done_idxs: 
        if done_idx == -1: 
            rtgs[start_index: start_index + 30] = -100 #TODO: if not finish the episode, set the reward to -100 
        else: 
            rtgs[start_index: start_index + done_idx + 1] = np.cumsum(rewards[start_index: start_index + done_idx + 1][::-1])[::-1]
        start_index += done_idx + 1
        
    assert len(rtgs) == len(rewards), "Error: length of RTG and reward are not equal"
    print("RTGS Length: ",len(rtgs), "Rewards Length: ", len(rewards))
            
    
    rtgs = np.array(rtgs)
    print('max rtg is %d' % max(rtgs))
    
    # -- create timesteps dataset 
    start_index = 0 
    time_steps = np.zeros(len(rewards), dtype=int)
    for done_idx in done_idxs: 
        if done_idx == -1: 
            time_steps[start_index: start_index + 30] = np.arange(0, 30)
        else: 
            insert = np.arange(0, done_idx + 1)
            assert start_index + done_idx + 1 - start_index == len(insert), "Error: length of timesteps is not equal to done_idx"
            time_steps[start_index: start_index + done_idx + 1] = insert
        start_index += done_idx + 1
    print('max time step is %d' % max(time_steps))
    
    done_idxs = np.array(done_idxs)
    
    # TODO: Shall we need a done_idxs for whole sequences 
    done_idxs = np.cumsum(done_idxs + 1)
    
    
    return states, actions, targets, rtgs,  done_idxs, time_steps
    
    
if __name__ == '__main__':
    trajs = load_data('/home/dylan/projects/motion_hcl/Matterport3DSimulator/tasks/R2R/trajs')
    states, actions, targets, rtgs,  done_idxs, time_steps = create_dataset(trajs)
    dataset = StateActionReturnDataset(states, 5 * 3, actions, targets, done_idxs, rtgs, time_steps)
    
    # test the dataset 
    try: 
        try_data = dataset[0]
    except: 
        raise NotImplementedError()
    
    # Now train the model 
    dataset = StateActionReturnDataset(states, 5 * 3, actions, targets, done_idxs, rtgs, time_steps)
    sub_train = torch.utils.data.Subset(dataset, list(range(100)))
    sub_test = torch.utils.data.Subset(dataset, list(range(100, 110)))
    

    mconf = GPT1Config(dataset.vocab_size, dataset.block_size,
                     model_type=args.model_type, max_timestep=max(time_steps))
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    epochs = args.epochs
    tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(sub_train)*args.context_length*3,
                        num_workers=4, seed=args.seed, model_type=args.model_type, max_timestep=max(time_steps))
    trainer = Trainer(model, sub_train, sub_test, tconf)

    trainer.train() 
    
    model = trainer.get_trained_model()
    
    # TODO: change this 
    
    xs = []
    for it, (x, y, target, rtg, timestep) in enumerate(sub_test):
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        target = target.to(trainer.device)
        rtg = rtg.to(trainer.device)
        timestep = timestep.to(trainer.device)
        # TODO: Sliding Windows Prediction
        predict_action = sample(model, x.unsqueeze(0), 5, temperature=1.0, sample=True, top_k=5, actions=y.unsqueeze(0), rtgs=rtg.unsqueeze(0), timesteps=timestep.unsqueeze(0)) #5 个上下文窗口预测未来一个 action
        y[:, -2:-1, :] = predict_action
        
        