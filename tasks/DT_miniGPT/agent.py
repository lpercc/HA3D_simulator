''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import random
import time
import numpy as np
import torch
from utils import check_agent_status
from reward import RewardCalculater
from tqdm import tqdm
import gc
from param import args


class BaseAgent(object):
    ''' Base class for an HA agent to generate and save trajectories. '''

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
            'state_features': [],
            'final_reward': [], 
            'sparse_final_reward': [],
            'target_reward': [],
            'path_reward': [],
            'missing_reward': [],
            'human_reward': [],
            'crashed': [],
        } for ob in obs]
        # self.steps = random.sample(range(-11,1), len(obs))
        self.steps = np.random.randint(-11, 1, size=len(obs))# ramdom from -11 - 0 (12 numbers) to choose the direction, because we have 12 discrete views

        ended = [False] * len(obs) # Is this enough for us to get a random walk agents?
        turn_right = [random.choice([True, False]) for _ in range(len(obs))]
        last_distances = [ob['distance'] for ob in obs]
        for _, t in enumerate(range(max_steps)): # 30 Steps 之后所有 Agent 的状态
            actions = []
            for i,ob in enumerate(obs):
                #TODO - 5 改为 随机
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
                    turn_right[i] = random.choice([True, False])
                else:
                    if turn_right[i]:
                        actions.append((0, 1, 0))
                    else:
                        actions.append((0, -1, 0))
                # turn right/left until we can go forward
            
                # 对于 traj 来说, 真正有用的 Actions 在 steps == 0 之后的 actions, 因为在这之前都是在调整方向
                # only need to record once after the end of navigation
                
                if self.steps[i] > 6: 
                    record_end_flag = True
                else:
                    record_end_flag = False

                if self.steps[i] >= 0 and not record_end_flag:
                    delta_distance = ob['distance'] - last_distances[i] if t > 0 else 0
                    last_distances[i] = ob['distance']
                    rwdclters = RewardCalculater()
                    is_crashed = ob['isCrashed']
                    distance = ob['distance']
                    rwdclters._set_ob(distance, is_crashed, actions[i], delta_distance)
                    reward = rwdclters.calculate(reward_type='dense')

                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    traj[i]['teacher_actions'].append(get_indexed_teacher_action(ob['teacher']))
                    traj[i]['student_actions'].append(get_indexed_teacher_action(actions[i]))
                    # TODO: Teacher action rewards
                    traj[i]['state_features'].append(ob['state_features'])
                    traj[i]['crashed'].append(ob['isCrashed'])
                    traj[i]['final_reward'].append(reward[0])
                    traj[i]['target_reward'].append(reward[1])
                    traj[i]['path_reward'].append(reward[2])
                    traj[i]['missing_reward'].append(reward[3])
                    traj[i]['human_reward'].append(reward[4])
        
                    
                    if len(traj[i]['unique_path']) == 0 or ob['viewpoint'] != traj[i]['unique_path'][-1]:
                        traj[i]['unique_path'].append(ob['viewpoint'])
                    assert traj[i]['target_reward'][0]['reward_strategy_1'] == 0
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
            'state_features': [],
            'final_reward': [], 
            'sparse_final_reward': [],
            'target_reward': [],
            'path_reward': [],
            'missing_reward': [],
            'human_reward': [],
            'actions': [],
            'crashed': [],
        } for ob in obs]
        ended = [False] * len(obs)
        last_distances = [ob['distance'] for ob in obs]
        for _, t in enumerate(range(max_steps)):  # Execute steps until max_steps or all agents have ended
            actions = [ob['teacher'] for ob in obs]  # Follow the teacher's action
            for i, ob in enumerate(obs):
                if not ended[i]:
                    delta_distance = ob['distance'] - last_distances[i] if t > 0 else 0 
                    last_distances[i] = ob['distance']
                    rwdclters = RewardCalculater()
                    is_crashed = ob['isCrashed']
                    distance = ob['distance']
                    rwdclters._set_ob(distance, is_crashed, actions[i], delta_distance)
                    reward = rwdclters.calculate(reward_type='dense')
                    # Append additional information to trajectory
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    traj[i]['teacher_actions'].append(get_indexed_teacher_action(ob['teacher']))
                    traj[i]['student_actions'].append(get_indexed_teacher_action(actions[i]))
                    # TODO: Calculate and append teacher action rewards
                    traj[i]['state_features'].append(ob['state_features'])
                    traj[i]['crashed'].append(ob['isCrashed'])
                    traj[i]['final_reward'].append(reward[0])
                    traj[i]['target_reward'].append(reward[1])
                    traj[i]['path_reward'].append(reward[2])
                    traj[i]['missing_reward'].append(reward[3])
                    traj[i]['human_reward'].append(reward[4])
                    #print(traj[i]['target_reward'][0]['reward_strategy_1'], traj[i]['target_reward'][t]['reward_strategy_1'])
                    assert traj[i]['target_reward'][0]['reward_strategy_1'] == 0 or t == 0
                    # add sparse_final_reward
                    if len(traj[i]['unique_path']) == 0 or ob['viewpoint'] != traj[i]['unique_path'][-1]:
                        traj[i]['unique_path'].append(ob['viewpoint'])
                
                if actions[i] == (0, 0, 0):  # If the action is to stop
                    ended[i] = True
                
            # Early exit if all agents have ended
            if all(ended):
                break
            
            obs = self.env.step(actions)
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
    reward_strategy = ''
    def __init__(self, env, results_path, model):
        super().__init__(env, results_path)
        self.model = model # init our DT here. trained model.
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_steps = args.max_episode_len # max run 30 steps
        

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
            target_returns = np.ones((len(obs), self.max_steps, 1)) * args.target_rtg # TODO: add to global config
            timesteps = np.zeros((len(obs), self.max_steps, 1), dtype=np.int64) # set the timesteps to 0
            timesteps = np.tile(np.arange(self.max_steps).reshape(1, -1, 1), (len(obs), 1, 1))
            actions = np.zeros((len(obs), self.max_steps, 1)) # set first time is zero for action,

        return states, actions, target_returns, timesteps
    
    def _check_action_valid(self, ob):
        ''' if a action is go forward, check if the agent can go forward. If not, resample the action until the agent can go forward.
        '''
        
        if len(ob['navigableLocations']) >= 2:  # 设置为大于等于 2, 因为自身也被算作 navigableLocations
            return True 
        else: 
            return False
    
    def set_reward(self, reward_strategy):
        self.reward_strategy =reward_strategy

    @torch.no_grad()
    def rollout(self):
        obs = np.array(self.env.reset())
        states, actions, target_returns, timesteps = self._variable_from_obs(obs, True)
        print(f"target returns: {target_returns}")
        batch_size = len(obs)
        ended = [False for _ in range(batch_size)]
        ended_set = set()
        
        
        
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'], ob['isCrashed'])]
        } for ob in obs] # do not need perm obs here, because we do not use curriculum learning. Actaully we do not train model.

        
        # Initialize lists to accumulate data for concatenation after the loop
        pbar = tqdm(enumerate(range(self.max_steps)), total=self.max_steps) # TODO: change this to config as max steps
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
                        next_action = self.model.sample_from_logits(next_actions_logit, temperature=1.0, sample=False, top_k=None) # size (batch_size, 1)
                        # check if the action is valid 
                        # only when the action is forward, we need to check if the agent can go forward.
                        can_forward = self._check_action_valid(ob)
                        while next_action[0].item() == 5 and not can_forward: # NOTE: must set with sample = True, because we need to sample the action again.
                            next_actions_logit[:, 5] = -float('inf') # set the forward action to -inf   
                            next_action = self.model.sample_from_logits(next_actions_logit, temperature=1.0, sample=True, top_k=None) #TODO: chang to sample from no main distribution.
                        
                        actions[i:i+1, 0, :] = next_action.unsqueeze(1).cpu().detach().numpy() # set the action to the last action
                        if next_action[0].item() == 4: # if the action is stop, we need to set the ended flag to True
                            ended[i] = True
                    else: # if not the first time, we predict as teacher force way then padding to max length
                        predict_action = self.model.get_action_prediction(state_t, actions=action_t, rtgs=target_return_t, timesteps=timestep_t)
                        next_actions_logit = predict_action[:, -1, :] # shape (batch_size, 1, n_actions)
                        # here, model predcit the logits, we need to sample the action from the logits.
                        #NOTE - Set sample and top_k = False, then we choose top1 action
                        next_action = self.model.sample_from_logits(next_actions_logit, temperature=1.0, sample=False, top_k=None)
                        can_forward = self._check_action_valid(ob)
                        while next_action[0] == 5 and not can_forward: # NOTE: must set with sample = True, because we need to sample the action again.
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
                distance = ob['distance']
                is_crashed = ob['isCrashed']
                rwdclter = RewardCalculater()
                rwdclter._set_ob(distance, is_crashed, actions_to_env[i], delta_distance)
                reward = rwdclter.calculate()
                #NOTE: 此处和 Reward 的策略要一致, [0] is for final reward
                next_rewards[i][0][0] = reward[0][self.reward_strategy] 
            # updated stuffs should add to sequence 
            # # DONE: change to modify a numpy array in place instead of concatenate
            if step < self.max_steps - 1:
                states[:, step+1, :] = new_states.reshape(batch_size, -1)
                target_returns[:, step+1, :] = target_returns[:, step, :].reshape(batch_size, -1) - next_rewards.reshape(batch_size, -1)

            for i, ob in enumerate(obs):
                # 如果观察已结束，并且还没有记录过结束信息，则记录
                #TODO - 改成方便 Json 解析的形式
                if ended[i] and i not in ended_set:
                    print(f'ob{i} ended, recorded')
                    #FIXME - traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation'], ob['isCrashed']))
                    ended_set.add(i) # 将结束的观察添加到集合中
                # 如果观察还没有结束，则记录其路径信息
                elif not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation'], ob['isCrashed']))
            
            if len(ended_set) == len(obs):
                print('All ended. Dropping out of loop')
                break
            # del unused variables         
            gc.collect()
            
        return traj

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
