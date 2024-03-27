'''
Author: Dylan Li dylan.h.li@outlook.com
Date: 2024-03-26 15:31:50
LastEditors: Dylan Li dylan.h.li@outlook.com
LastEditTime: 2024-03-26 20:36:59
FilePath: /Matterport3DSimulator/tasks/R2R/DT/train_GPT.py
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
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
args = parser.parse_args()

seed_everything(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size):        
        """ This is a dataset class for the State-Action-Reward dataset.
        How do we model a GPT like model here? For data, we concanate all text into one list than split it by context windows. 

        Args:
            data (list[np.arrays]): should be a list of state_features 
            block_size (int): The is real transformer context length . block_size // 3 will be the input context length for actions, rewards and so on
            actions (list[int]): actions should be a list of integers.
            done_idxs (list[int]): A list of indices where the episode ends. 
            rtgs (list): A list of return to go rewards.
            timesteps (list): A list of timesteps
            
        return: 
            states
            actions
            rtgs
            timesteps 
        """
        self.block_size = block_size
        # self.vocab_size = max(actions) + 1
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size # final block will not be counted 

    def __getitem__(self, idx):
        block_size = self.block_size // 3 

        raise  NotImplementedError("Subclasses must implement this method.")
    
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
    
    
    return states, actions, rtgs, actions, targets, done_idxs
    
    
if __name__ == '__main__':
    trajs = load_data('/home/dylan/projects/motion_hcl/Matterport3DSimulator/tasks/R2R/trajs')
    states, actions, rtgs, actions, targets, done_idx = create_dataset(trajs)