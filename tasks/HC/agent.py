''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from env import HCBatch
from torch import optim
from torch.autograd import Variable
from utils import check_agent_status, padding_idx, RewardCalculater
from tqdm import tqdm
import gc


class BaseAgent(object):
    ''' Base class for an HC agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        if not os.path.exists(os.path.dirname(self.results_path)):
            os.makedirs(os.path.dirname(self.results_path))

        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print('Testing %s' % self.__class__.__name__)
        looped = False
        while True:
            for traj in self.rollout():
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break
    
    def test_teacher(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print('Testing %s' % self.__class__.__name__)
        looped = False
        while True:
            for traj in self.teacher_rollout():
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break    


class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self):
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in self.env.reset()]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''
        
        # five viewpoint steps and then stops. Total 6 viewpoints. 
        # choose five because the maximum number of steps is 5 to get to the goal.
    def rollout(self, max_steps=30):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [],
            'teacher_actions': [],
            'student_actions': [],
            'unique_path': [ob['viewpoint']],
            'teacher': [],
            'state_features': [],
            'final_reward': [], 
            'target_reward': [],
            'path_reward': [],
            'missing_reward': [],
            'human_reward': [],
            'actions': [],
            'crashed': [],
        } for ob in obs]
        # self.steps = random.sample(range(-11,1), len(obs))
        self.steps = np.random.randint(-11, 1, size=len(obs))# ramdom from -11 - 0 (12 numbers) to choose the direction, because we have 12 discrete views

        ended = [False] * len(obs) # Is this enough for us to get a random walk agents?
        for _, t in enumerate(range(max_steps)): # 30 Steps 之后所有 Agent 的状态
            actions = []
            last_distances = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5: # End of navigation larger than 5 steps (including the first one) 
                    actions.append((0, 0, 0))# do nothing, i.e. end
                    ended[i] = True
                    self.steps[i] += 1
                elif self.steps[i] < 0: # 等价于随机起始一个方向, 直到 steps[i] == 0
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else:
                    actions.append((0, 1, 0))
                # turn right until we can go forward

                last_distances.append(ob['distance'])
            
                # 对于 traj 来说, 真正有用的 Actions 在 steps == 0 之后的 actions, 因为在这之前都是在调整方向
                # only need to record once after the end of navigation
                
                if self.steps[i] > 6: 
                    record_end_flag = True
                else:
                    record_end_flag = False

                if self.steps[i] >= 0 and not record_end_flag:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    traj[i]['teacher_actions'].append(get_indexed_teacher_action(ob['teacher']))
                    traj[i]['student_actions'].append(get_indexed_teacher_action(actions[i]))
                    # TODO: Teacher action rewards
                    traj[i]['state_features'].append(ob['state_features'])
                    traj[i]['crashed'].append(ob['isCrashed'])
                    
                    delta_distance = ob['distance'] - last_distances[i] if t > 0 else 0
                    rwdclter = RewardCalculater()
                    rwdclter._set_ob(ob, actions[i], delta_distance)
                    reward = rwdclter.calculate()
                    traj[i]['final_reward'].append(reward[0])
                    traj[i]['target_reward'].append(reward[1])
                    traj[i]['path_reward'].append(reward[2])
                    traj[i]['missing_reward'].append(reward[3])
                    traj[i]['human_reward'].append(reward[4])
                    
                    if len(traj[i]['unique_path']) == 0 or ob['viewpoint'] != traj[i]['unique_path'][-1]:
                        traj[i]['unique_path'].append(ob['viewpoint'])
    
            obs = self.env.step(actions)
        # Check Agent 
        #check_agent_status(traj, max_steps, ended)
                        
        # calculate all reward here , for quick test, new we use just to find the shortest path 
        # for each step , we calculate the distances between the current viewpoint and the goal. 
        

        return traj
    

class TeacherAgent(BaseAgent):
    ''' An agent that follows the teacher's actions exactly. '''

    def rollout(self, max_steps=30):
        """
        Executes a rollout in the environment using the teacher's actions for guidance.

        This method follows the teacher's actions for each observation in the environment
        up to a maximum number of steps. It collects the trajectory of each agent, including
        the path taken, actions, rewards, and other relevant information. The method is designed
        to simulate the agent's behavior in the environment and collect data for analysis or
        further training.

        Parameters:
        - max_steps (int): The maximum number of steps to execute for each agent in the rollout.
                            Default is 30 steps.

        Returns:
        - list[dict]: A list of dictionaries, each representing the trajectory of an agent.
                        Each dictionary contains keys such as 'instr_id', 'path', 'teacher_actions',
                        'student_actions', 'state_features', 'final_reward', 'target_reward',
                        'path_reward', 'missing_reward', 'human_reward', 'actions', and 'crashed',
                        detailing the respective aspects of the agent's trajectory.
        """
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [],
            'teacher_actions': [],
            'student_actions': [],
            'unique_path': [ob['viewpoint']],
            'teacher': [],
            'state_features': [],
            'final_reward': [], 
            'target_reward': [],
            'path_reward': [],
            'missing_reward': [],
            'human_reward': [],
            'actions': [],
            'crashed': [],
        } for ob in obs]

        ended = [False] * len(obs)
        for _, t in enumerate(range(max_steps)):  # Execute steps until max_steps or all agents have ended
            actions = [ob['teacher'] for ob in obs]  # Follow the teacher's action
            last_distances = []
            for i, ob in enumerate(obs):
                if actions[i] == (0, 0, 0):  # If the action is to stop
                    ended[i] = True
                last_distances.append(ob['distance'])
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    # Append additional information to trajectory
                    traj[i]['teacher_actions'].append(get_indexed_teacher_action(ob['teacher']))
                    traj[i]['student_actions'].append(get_indexed_teacher_action(actions[i]))
                    # TODO: Calculate and append teacher action rewards
                    traj[i]['state_features'].append(ob['state_features'])
                    traj[i]['crashed'].append(ob['isCrashed'])
                    
                    delta_distance = ob['distance'] - last_distances[i] if t > 0 else 0
                    rwdclter = RewardCalculater()
                    rwdclter._set_ob(ob, actions[i], delta_distance)
                    reward = rwdclter.calculate()
                    traj[i]['final_reward'].append(reward[0])
                    traj[i]['target_reward'].append(reward[1])
                    traj[i]['path_reward'].append(reward[2])
                    traj[i]['missing_reward'].append(reward[3])
                    traj[i]['human_reward'].append(reward[4])
                    
                    if len(traj[i]['unique_path']) == 0 or ob['viewpoint'] != traj[i]['unique_path'][-1]:
                        traj[i]['unique_path'].append(ob['viewpoint'])
            # Early exit if all agents have ended
            if all(ended):
                break
            obs = self.env.step(actions)
        return traj

class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))
        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = self.env.step(actions)
            for i,a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj


class DecisionTransformerAgent(BaseAgent):
    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
      (0,-1, 0), # left
      (0, 1, 0), # right
      (0, 0, 1), # up
      (0, 0,-1), # down
      (1, 0, 0), # forward
      (0, 0, 0), # <end>
      (0, 0, 0), # <start>
      (0, 0, 0)  # <ignore>
    ]
    
    indexed_to_teacher_action = {
        4: (0, 0, 0), # stop
        0: (0, 1, 0), # turn right
        1: (0, -1, 0), # turn left
        2: (0, 0, 1), # move up
        3: (0, 0, -1), # move down
        5: (1, 0, 0), # forward
    }
    
    def __init__(self, env, results_path, model):
        super().__init__(env, results_path)
        self.model = model # init our DT here. trained model.
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_steps = 30 # max run 30 steps
        

    def _variable_from_obs(self, obs, return_whole=False):
        ''' Extracts the feature tensor from a list of observations. 
        - obs[np.array]: a list of observations.
        '''
        
        states = []
        
        for ob in obs: 
            state = ob['state_features']
            states.append(state)
        
        
        if not return_whole:    
            states = np.array(states, dtype=np.float32).reshape(len(obs), 1, -1)# (batch_size, 1, feature_size)
            target_returns = np.ones((len(obs), 1, 1)) * 0.5 # set the target return to 3.0, because we have sparse positive reward
            timesteps = np.zeros((len(obs), 1, 1), dtype=np.int64) # set the timesteps to 0
            actions = np.zeros((len(obs), 1, 1)) - 1 # set first time is None for action
        else: 
            states = np.repeat(np.array(states, dtype=np.float32).reshape(len(obs), 1, -1), self.max_steps, axis=1)
            target_returns = np.ones((len(obs), self.max_steps, 1)) * 3 # set the target return to 3.0, because we have sparse positive reward
            timesteps = np.zeros((len(obs), self.max_steps, 1), dtype=np.int64) # set the timesteps to 0
            timesteps = np.tile(np.arange(self.max_steps).reshape(1, -1, 1), (len(obs), 1, 1))
            actions = np.zeros((len(obs), self.max_steps, 1)) # set first time is zero for action,

        return states, actions, target_returns, timesteps
    
    def _check_action_valid(self, ob):
        ''' if a action is go forward, check if the agent can go forward. If not, resample the action until the agent can go forward.
        - next_action: the action that the model predict. size (batch_size, 1). Batch size should be 1.
        '''
        if len(ob['navigableLocations']) > 1:
            return True 
        else: 
            return False

    @torch.no_grad()
    def rollout(self):
        obs = np.array(self.env.reset())
        states, actions, target_returns, timesteps = self._variable_from_obs(obs, True)
        batch_size = len(obs)
        ended = [False for _ in range(batch_size)]
        
        
        
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'], ob['isCrashed'])]
        } for ob in obs] # do not need perm obs here, because we do not use curriculum learning. Actaully we do not train model.

        
        # Initialize lists to accumulate data for concatenation after the loop
        pbar = tqdm(enumerate(range(self.max_steps)), total=30) # TODO: change this to config as max steps
        for _, step in pbar: # max run 30 steps
            pbar.set_description(f"Step {step}")
            # NOTE: can not set batch here
            actions_to_env = []
            last_distance = []
            
            for i, ob in enumerate(obs):
                # get the state, action, target_return, timestep for this observation
                state = states[i:i+1, :step + 1, :] # size (1, step + 1, feature_size)
                action = actions[i:i+1, :step + 1, :]  # size (1, step + 1, 1)
                target_return = target_returns[i:i+1, :step + 1, :]  # size (1, step + 1, 1)
                timestep = timesteps[i:i+1, :step + 1, :]  # size (1, step + 1, 1)

                # inference the model
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
                    action_t = torch.tensor(action, dtype=torch.long).to(self.device)
                    target_return_t = torch.tensor(target_return, dtype=torch.float32).to(self.device)
                    timestep_t = torch.tensor(timestep, dtype=torch.int64).to(self.device)
                    
                    # if the first time. we need to set the action to None, because it can not exceed the max length of the sequence, so we can only use forward. 
                    if step == 0: 
                        predict_action , _ = self.model.forward(state_t, actions=None, targets=None, rtgs=target_return_t, timesteps=timestep_t)
                        next_actions_logit = predict_action[:, -1, :] # shape (batch_size, n_actions)
                        # here, model predcit the logits, we need to sample the action from the logits.
                        next_action = self.model.sample_from_logits(next_actions_logit, temperature=1.0, sample=True, top_k=1) # size (batch_size, 1)
                        # check if the action is valid 
                        # only when the action is forward, we need to check if the agent can go forward.
                        while next_action[0].item() == 5 and not self._check_action_valid(ob): # NOTE: must set with sample = True, because we need to sample the action again.
                            next_actions_logit[:, 5] = -float('inf') # set the forward action to -inf   
                            next_action = self.model.sample_from_logits(next_actions_logit, temperature=1.0, sample=True, top_k=None) #TODO: chang to sample from no main distribution.
                        
                        actions[i:i+1, 0, :] = next_action.unsqueeze(1).cpu().detach().numpy() # set the action to the last action
                        if next_action[0].item() == 4: # if the action is stop, we need to set the ended flag to True
                            ended[i] = True
                    else: # if not the first time, we predict as teacher force way then padding to max length
                        predict_action = self.model.get_action_prediction(state_t, actions=action_t, rtgs=target_return_t, timesteps=timestep_t)
                        next_actions_logit = predict_action[:, -1, :] # shape (batch_size, 1, n_actions)
                        # here, model predcit the logits, we need to sample the action from the logits.
                        next_action = self.model.sample_from_logits(next_actions_logit, temperature=1.0, sample=True, top_k=None)
                        while next_action[0] == 5 and not self._check_action_valid(ob): # NOTE: must set with sample = True, because we need to sample the action again.
                            next_actions_logit[:, 5] = -float('inf') # set the forward action to -inf 
                            next_action = self.model.sample_from_logits(next_actions_logit, temperature=1.0, sample=True, top_k=None)
                        actions[i:i+1, step, :] = next_action.unsqueeze(1).cpu().detach().numpy() # set the action to the last action
                        if next_action[0].item() == 4:
                            ended[i] = True
                    
                # convert the action to the env action
                action_to_env = self.indexed_to_teacher_action[next_action[0].item()]
                actions_to_env.append(action_to_env)
                
                # save last distance
                last_distance.append(ob['distance'])
            
            # Now we can interact with the environment
            obs = self.env.step(actions_to_env)
            
            # get new states, actions, target_returns, timesteps
            new_states, _, _, _ = self._variable_from_obs(obs)
        
            # calculate next rewards
            next_rewards = np.zeros((batch_size, 1, 1))

            for i, ob in enumerate(obs):
                delta_distance = ob['distance'] - last_distance[i] if step > 0 else 0
                rwdclter = RewardCalculater()
                rwdclter._set_ob(ob, actions[i], delta_distance)
                reward = rwdclter.calculate()
                next_rewards[i][0][0] = reward['reward_strategy_1'][0]
            
            # updated stuffs should add to sequence 
            # # DONE: change to modify a numpy array in place instead of concatenate
            if step < self.max_steps - 1:
                states[:, step+1, :] = new_states.reshape(batch_size, -1)
                target_returns[:, step+1, :] = target_returns[:, step, :].reshape(batch_size, -1) - next_rewards.reshape(batch_size, -1)

            
            # update trajs
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation'], ob['isCrashed']))
            
            
            # del unused variables         
            gc.collect()
            
        return traj


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
      (0,-1, 0), # left
      (0, 1, 0), # right
      (0, 0, 1), # up
      (0, 0,-1), # down
      (1, 0, 0), # forward
      (0, 0, 0), # <end>
      (0, 0, 0), # <start>
      (0, 0, 0)  # <ignore>
    ]
    feedback_options = ['teacher', 'argmax', 'sample']
    
    

    def __init__(self, env, results_path, encoder, decoder, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.model_actions.index('<ignore>'))

    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False).cuda()

    def teacher_rollout(self):
        obs = np.array(self.env.reset())
        batch_size = len(obs)
        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'], ob['isCrashed'])]
        } for ob in perm_obs]
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env
        env_action = [None] * batch_size
        for t in range(self.episode_len):

            # Supervised training
            target = self._teacher_action(perm_obs, ended)


            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = target[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation'], ob['isCrashed']))

            # Early exit if all ended
            if ended.all():
                break
        return traj

    def rollout(self):
        obs = np.array(self.env.reset())
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'], ob['isCrashed'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        for t in range(self.episode_len):

            f_t = self._feature_variable(perm_obs) # Image features from obs
            h_t,c_t,alpha,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation'], ob['isCrashed']))

            # Early exit if all ended
            if ended.all():
                break

        self.losses.append(self.loss.item() / self.episode_len)
        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqAgent, self).test()
    
    def test_teacher(self):
        ''' Evaluate once on each instruction in the current environment '''
        super(Seq2SeqAgent, self).test_teacher()
    
    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.rollout()
            self.loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))


def get_indexed_teacher_action(teacher_action):
    # TODO: change name and move to utils
    if teacher_action == (0, 0, 0):
        return 4 #stop
    elif teacher_action == (0, 1, 0): # turn right 
        return 0
    elif teacher_action == (0, -1, 0): # turn left
        return 1
    elif teacher_action == (0, 0, 1):
        return 2
    elif teacher_action == (0, 0, -1):
        return 3
    else:
        return 5 #forward
