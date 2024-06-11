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
from GPT_trainer import Trainer
from utils import seed_everything
from collections import deque
import random
import torch
import os 
import pickle
import gzip
#import blosc
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from param import args
HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")
print(f"sim root dir:{HA3D_SIMULATOR_PATH}")
MODEL_DIR = os.path.join(HA3D_SIMULATOR_PATH, "tasks/DT_miniGPT/models")
TRAJS_DIR = os.path.join(HA3D_SIMULATOR_PATH, "tasks/DT_miniGPT/trajs")

# Iterate through the args dictionary and print each entry's key (parameter name) and value
for key, value in vars(args).items():
    print(f"{key} = {value}")
user_input = input("Confirm parameters are correct and continue with the program? (yes/no): ").strip().lower()  # Get user input, remove leading and trailing spaces, convert to lowercase

if user_input == "yes":
    print("Continuing program execution...")
    # Add code here to continue program execution if needed
elif user_input == "no":
    print("Program terminated.")
    exit()  # Use exit() function to terminate the program
else:
    print("Invalid input, please enter yes or no.")
    exit()
    # You may choose to ask again or terminate the program

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, targets, done_idxs, rtgs, timesteps):        
        """ This is a dataset class for the State-Action-Reward dataset.
        How do we model a GPT like model here? For data, we concanate all text into one list than split it by context windows. 

        Args:
            data (np.arryas): should be a list of state_features 
            block_size (arryas): The is real transformer context length . block_size // 3 will be the input context length for actions, rewards and so on
            actions (arryas]): actions should be a list of integers.
            done_idxs (list[int]): A list of indices where the episode ends. 
            rtgs (arryas): A list of return totasks/DT/models/ go rewards.
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
        timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64).unsqueeze(1) # (block_size, 1)
        
        return states, actions, targets, rtgs, timesteps
    
def load_data(data_dir, trajs_type): 
    # TODO: Train as incremental learning
    train_trajs = []
    val_seen_trajs = []
    val_unseen_trajs = []
    trajs_type = trajs_type.split('_')
    for traj_type in trajs_type:
        #NOTE - Here load all trajs including teacher and random
        with open(os.path.join(data_dir, f"train_trajs_{traj_type}_{args.dataset_name}.pkl"), 'rb') as f: #DONE: change to support pkl 
            train_traj = pickle.load(f) # 
            if args.mode == 'debug':
                train_trajs.extend(train_traj[:args.train_samples])
            else:
                train_trajs.extend(train_traj)

        with open(os.path.join(data_dir, f"val_seen_trajs_{traj_type}_{args.dataset_name}.pkl"), 'rb') as f: #DONE: change to support pkl 
            val_seen_traj = pickle.load(f) # 
            if args.mode == 'debug':
                val_seen_trajs.extend(val_seen_traj[:args.train_samples])
            else:
                val_seen_trajs.extend(val_seen_traj)
        
        with open(os.path.join(data_dir, f"val_unseen_trajs_{traj_type}_{args.dataset_name}.pkl"), 'rb') as f: #DONE: change to support pkl 
            val_unseen_traj = pickle.load(f) # 
            if args.mode == 'debug':
                val_unseen_trajs.extend(val_unseen_traj[:args.train_samples])
            else:
                val_unseen_trajs.extend(val_unseen_traj)
    
    print(f"{trajs_type} trajs data len train={len(train_trajs)} val_seen={len(val_seen_trajs)} val_unseen={len(val_unseen_trajs)}")
    return train_trajs, val_seen_trajs, val_unseen_trajs

def create_dataset(trajs,reward_strategy):
    # This is a function to read all trajs into one big dataset. 
    states = [] 
    actions = []
    rewards = [] 
    actions = [] 
    targets =  []
    done_idxs = []
    for t in trajs: 
        states.extend(t['state_features'])
        actions.extend(t['student_actions']) #TODO： 和其对齐
        rewards.extend(t['final_reward'])
        targets.extend(t['teacher_actions'])
        done_idxs.append(len(t['student_actions']) - 1) # -1 because the index starts from 0
    
    # Convert to numpy arrays
    states  = np.array(states)
    targets = np.array(targets)
    rewards = [reward_dict[reward_strategy] for reward_dict in rewards]
    rewards = np.array(rewards)
    actions = np.array(actions)
    print(f"states shape:{states.shape}, reward shape:{rewards.shape}   ")
    assert np.sum(done_idxs) == len(actions) - len(done_idxs), "Error: sum of done_idxs is not equal to length of actions"
    
    
    # -- create return to go reward datasets TODO: return 是错位的
    rtgs = np.zeros(len(rewards))
    start_index = 0
    for done_idx in done_idxs: 
        if done_idx == -1: 
            rtgs[start_index: start_index + args.max_episode_len] = -100 #FIXME: if not finish the episode, set the reward to -100 
        else: 
            rtgs[start_index: start_index + done_idx + 1] = np.cumsum(rewards[start_index: start_index + done_idx + 1][::-1])[::-1]
        start_index += done_idx + 1
        
    assert len(rtgs) == len(rewards), "Error: length of RTG and reward are not equal"
    print("RTGS Length: ",len(rtgs), "Rewards Length: ", len(rewards))
            
    
    rtgs = np.array(rtgs)
    print('max rtg is %d' % max(rtgs))
    print('min rtg is %d' % min(rtgs))
    
    # -- create timesteps dataset 
    start_index = 0 
    time_steps = np.zeros(len(rewards), dtype=int)
    for done_idx in done_idxs: 
        if done_idx == -1: 
            time_steps[start_index: start_index + args.max_episode_len] = np.arange(0, args.max_episode_len)
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
    model_dir = os.path.join(MODEL_DIR, f'{args.experiment_id}_{args.model_name}_{args.feedback_method}_{args.reward_strategy}')
    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(TRAJS_DIR):
        os.makedirs(TRAJS_DIR)


    seed_everything(args.seed)
    train_trajs, val_seen_trajs, val_unseen_trajs = load_data(TRAJS_DIR, args.feedback_method)
    print(f"train_trajs len:{len(train_trajs)}")
    train_states, train_actions, train_targets, train_rtgs,  train_done_idxs, train_time_steps = create_dataset(train_trajs, args.reward_strategy)
    train_dataset = StateActionReturnDataset(train_states, 5 * 3, train_actions, train_targets, train_done_idxs, train_rtgs, train_time_steps)
    #TODO - Subset
    #indices = list(range(args.train_samples))
    #sub_train = torch.utils.data.Subset(train_dataset, indices)
    
    val_seen_states, val_seen_actions, val_seen_targets, val_seen_rtgs,  val_seen_done_idxs, val_seen_time_steps = create_dataset(val_seen_trajs, args.reward_strategy)
    val_seen_dataset = StateActionReturnDataset(val_seen_states, 5 * 3, val_seen_actions, val_seen_targets, val_seen_done_idxs, val_seen_rtgs, val_seen_time_steps)
    
    val_unseen_states, val_unseen_actions, val_unseen_targets, val_unseen_rtgs,  val_unseen_done_idxs, val_unseen_time_steps = create_dataset(val_unseen_trajs, args.reward_strategy)
    val_unseen_dataset = StateActionReturnDataset(val_unseen_states, 5 * 3, val_unseen_actions, val_unseen_targets, val_unseen_done_idxs, val_unseen_rtgs, val_unseen_time_steps)
    # test the dataset 
    try: 
        try_data = train_states[0]
    except: 
        raise NotImplementedError()

    mconf = GPT1Config(train_dataset.vocab_size, train_dataset.block_size,
                     model_type=args.model_type, max_timestep=max(train_time_steps))
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    #trainer = Trainer(model, sub_train, val_seen_dataset, val_unseen_dataset, args, model_dir)
    trainer = Trainer(model, train_dataset, val_seen_dataset, val_unseen_dataset, args, model_dir)


    if args.mode == 'val':
        for model_file in os.listdir(model_dir):
            if model_file.endswith(".pth"):
                cpt_path = os.path.join(model_dir, model_file)
                print(f"val mode {cpt_path}")
                trainer.load_checkpoint(cpt_path)
                eval_results, eval_results_dict = trainer.val()
                record_file = open(os.path.join(model_dir, "val_log.txt"), 'a')
                record_file.write(f"\nValidation model path:{cpt_path}\n{args}\n{eval_results}\n")
                record_file.close()
    elif args.mode == 'train': 
        # 先 Train
        trainer.train()
        # 开始 validation
        for model_file in os.listdir(model_dir):
            if model_file.endswith(".pth"):
                cpt_path = os.path.join(model_dir, model_file)
                print(f"val mode {cpt_path}")
                trainer.load_checkpoint(cpt_path)
                eval_results, eval_results_dict = trainer.val()
                record_file = open(os.path.join(model_dir, "val_log.txt"), 'a')
                record_file.write(f"\nValidation model path:{cpt_path}\n{args}\n{eval_results}\n")
                record_file.close()
    elif args.mode == 'debug': 
        model_save_path = os.path.join(model_dir, 'model_last.pth')
        record_file = open(os.path.join(model_dir, "train_log.txt"), 'a')
        record_file.write(f"\nTrain model path:{model_save_path}\n{args}\n")
        record_file.close()
        print(f"debug mode {model_save_path}")
        trainer.train()
        trainer.save_checkpoint(model_save_path)
    else:
        raise NotImplementedError()
    
    
    
