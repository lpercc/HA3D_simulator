import random
import numpy as np
import os
import sys
import json
import time
import math

class RewardCalculater():
    """
    A class to calculate and manage rewards for an agent based on its actions and state.

    Attributes:
        ob (dict): The current observation of the agent, containing environment and agent state.
        action (tuple): The action taken by the agent.
        delta_distance (float): The change in distance between the agent and the target.
    """

    def __init__(self):
        """
        Initializes the Reward object with default values.
        """
        self.action = tuple()
        self.delta_distance = float()
        self.isCrashed_record = [False]
        self.final_reward = {}
        self.target_reward = {}
        self.path_reward = {}
        self.miss_penalty = {}
        self.human_reward = {}
        
        
    def _set_ob(self, distance, is_crashed, action, delta_distance):
        """
        Sets the current observation, action, and delta distance for the reward calculation.

        Parameters:
            ob (dict): The current observation of the agent.
            action (tuple): The action taken by the agent.
            delta_distance (float): The change in distance between the agent and the target.
        """
        self.is_crashed = is_crashed
        self.distance = distance
        self.action = action
        self.delta_distance = delta_distance

    def calculate(self, reward_type='dense'):
        """
        Calculates the reward based on the current state, action, and reward strategy.

        Returns:
            tuple: A tuple containing the final reward, target reward, path reward, miss penalty, and human reward dict.
            such as dict final_reward = {
                'reward_strategy_1': float,
                'reward_strategy_2': float,
                'reward_strategy_3': float,
                'reward_strategy_4': float,
                'reward_strategy_5': float
            } 
        """

        self.reward_strategy_1(reward_type=reward_type)
        self.reward_strategy_2(reward_type=reward_type)
        self.reward_strategy_3(reward_type=reward_type)
        self.reward_strategy_4(reward_type=reward_type)
        self.reward_strategy_5(reward_type=reward_type)
        self.reward_strategy_6(reward_type=reward_type)
        
        return [self.final_reward, self.target_reward, self.path_reward, self.miss_penalty, self.human_reward]
    
    def append_rewards(self, final_reward, target_reward, path_reward, miss_penalty, human_reward, strategy_name):
        self.final_reward[strategy_name] = final_reward
        self.target_reward[strategy_name] = target_reward
        self.path_reward[strategy_name] = path_reward
        self.miss_penalty[strategy_name] = miss_penalty
        self.human_reward[strategy_name] = human_reward

    def get_final_reward(self, target_reward, path_reward, miss_penalty, human_reward, reward_type):
        # Determine the final reward based on the reward type
        # TODO: Depreated reward_type sparse
        if reward_type == 'sparse':
            final_reward = target_reward + human_reward
        elif reward_type == 'dense':
            final_reward = target_reward + path_reward + miss_penalty + human_reward
        return final_reward
    
    # 只设置在目标处的 Reward. 设计为 5 和 -5
    def reward_strategy_1(self, reward_type):
        target_reward = 0.0
        path_reward = 0.0
        miss_penalty = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method
        dist = self.distance
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 5.0
            else:
                target_reward = -5.0
        
        final_reward = self.get_final_reward(target_reward, path_reward, miss_penalty, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, miss_penalty, human_reward, strategy_name='reward_strategy_1')

    # 设置在目标处的 Reward + 避开人的 Reward. 碰到人给 -2
    def reward_strategy_2(self, reward_type):
        # NOTE: All compare to reward 1
        # Add human reward
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        step_reward = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method
        dist = self.distance
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 5.0
            else:
                target_reward = -5.0

        crashed = self.is_crashed
        if crashed:
            human_reward = -2

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_2')

    # 只设置 避开人的 Reward, -2
    def reward_strategy_3(self, reward_type):
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        step_reward = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method

        crashed = self.is_crashed
        if crashed:
            human_reward = -2

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_3')

    # 设置在目标处的 Reward + 靠近目标+1 + 远离-1 避开人的 Reward. 碰到人给 -2
    def reward_strategy_4(self, reward_type):
        # NOTE: All compare to reward 1
        # Add human reward
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        step_reward = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method
        dist = self.distance
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 5.0
            else:
                target_reward = -5.0
        
        delta_distance = self.delta_distance
        if delta_distance < 0:
            path_reward = 1.0
        elif delta_distance > 0:
            path_reward = -1.0

        crashed = self.is_crashed
        if crashed:
            human_reward = -2

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_4')
    # 设置在目标处的 Reward + 靠近目标+1 + 避开人的 Reward. 碰到人给 -2
    def reward_strategy_5(self, reward_type):
        # NOTE: All compare to reward 1
        # Add human reward
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        step_reward = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method
        dist = self.distance
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 5.0
            else:
                target_reward = -5.0
        
        delta_distance = self.delta_distance
        if delta_distance > 0:
            path_reward = -1.0

        crashed = self.is_crashed
        if crashed:
            human_reward = -2

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_5')

    def reward_strategy_6(self, reward_type):
        # NOTE: All compare to reward 1
        # Add human reward
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        step_reward = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method
        dist = self.distance
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 5.0
            else:
                target_reward = -5.0
        
        delta_distance = self.delta_distance
        if delta_distance > 0:
            path_reward = -1.0

        crashed = self.is_crashed
        if crashed:
            human_reward = -5

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_6')
