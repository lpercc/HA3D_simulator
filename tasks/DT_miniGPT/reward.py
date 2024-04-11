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
        self.ob = dict()
        self.action = tuple()
        self.delta_distance = float()
        self.isCrashed_record = [False]
        self.final_reward = {}
        self.target_reward = {}
        self.path_reward = {}
        self.miss_penalty = {}
        self.human_reward = {}
        
        
    def _set_ob(self, ob, action, delta_distance):
        """
        Sets the current observation, action, and delta distance for the reward calculation.

        Parameters:
            ob (dict): The current observation of the agent.
            action (tuple): The action taken by the agent.
            delta_distance (float): The change in distance between the agent and the target.
        """
        self.ob = ob
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
        dist = self.ob['distance']
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
        dist = self.ob['distance']
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 5.0
            else:
                target_reward = -5.0

        crashed = self.ob['isCrashed']
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

        crashed = self.ob['isCrashed']
        if crashed:
            human_reward = -2

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_3')

    # 
    def reward_strategy_4(self, consider_human, reward_type):
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        step_reward = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method

        # Get the current distance from the observation
        dist = self.ob['distance']
        epsilon = 0.1

        # Check if the agent has stopped
        if consider_human:
            # Check if the agent has crashed
            crashed = self.ob['isCrashed']
            if crashed:
                if self.isCrashed_record[-1]:
                    human_reward = -2.0  # Negative reward for crashing
                else:
                    human_reward = -4.0  # Negative reward for crashing
            else:
                if self.isCrashed_record[-1]:
                    human_reward = 12/(dist + epsilon)  # Negative reward for crashing
                else:
                    human_reward = 6/(dist + epsilon)
                if self.delta_distance == 0:
                    human_reward = 0
            self.isCrashed_record.append(crashed)
        
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 3.0
            else:
                target_reward = -3.0
        else:
            # Calculate path fidelity reward
            path_flag = - self.delta_distance
            if path_flag > 0.0:
                path_reward = 1
            elif path_flag < 0.0:
                path_reward = -1.0
            else:
                path_reward = -0.5

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_4')

    # 
    def reward_strategy_5(self, consider_human, reward_type):
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        step_reward = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method

        # Get the current distance from the observation
        dist = self.ob['distance']
        epsilon = 0.1

        # Check if the agent has stopped
        if consider_human:
            # Check if the agent has crashed
            crashed = self.ob['isCrashed']
            if crashed:
                if self.isCrashed_record[-1]:
                    human_reward = -2.0  # Negative reward for crashing
                else:
                    human_reward = -4.0  # Negative reward for crashing
            else:
                if self.isCrashed_record[-1]:
                    human_reward = 12/(dist + epsilon)  # Negative reward for crashing
                else:
                    human_reward = 6/(dist + epsilon)
                if self.delta_distance == 0:
                    human_reward = 0
            self.isCrashed_record.append(crashed)
        
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 3.0
            else:
                target_reward = -3.0
        else:
            # Calculate path fidelity reward
            path_flag = - self.delta_distance
            if path_flag > 0.0:
                path_reward = 1
            elif path_flag < 0.0:
                path_reward = -1.0
            else:
                path_reward = -0.01
            
        if self.action == (0, 0, 0):
            step_reward = 10 - self.ob['step']

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_5')


    def reward_strategy_6(self, consider_human, reward_type):
        """
        Description:
        The function calculates rewards for the agent based on its current observation, the action it took,
        the change in distance from its last location, and the type of reward it should receive.
        It considers various factors such as reaching the target, following the path, missing the target,
        and human interaction.

        The reward calculation is performed as follows:
        - If the agent has stopped (action is (0, 0, 0)):
            - If the agent is close to the target (distance < 3.0), a positive reward of 3.0 is given.
            - Otherwise, a negative reward of -3.0 is given.
        - If the agent is moving:
            - The path fidelity reward is calculated based on the change in distance (delta_distance).
                - If delta_distance > 0.0, the path reward is 0.0.
                - If delta_distance < 0.0, the path reward is -1.0. Which means the agent is moving away from the target. It encourages the agent to move towards the target.
                - If delta_distance = 0.0, a small negative reward of -0.1 is given to discourage staying in place.
            - The miss penalty is calculated based on the agent's distance from the target.
                - If the agent is close to the target (last_dist < 1.0) and moving away (delta_distance > 0.0),
                  the miss penalty is calculated as (last_dist - 1.0) * 2.0.
        - In non-test local environment, human interaction reward is calculated.
            - If the agent has crashed, a negative reward of -2.0 is given.
            - Otherwise, the human reward is 0.0.

        Returns:
            tuple: A tuple containing the target reward, path reward, miss penalty, and human reward.
        """
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        miss_penalty = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method

        # Get the current distance from the observation
        dist = self.ob['distance']

        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 3.0
            else:
                target_reward = -3.0
        else:
            # Calculate path fidelity reward
            path_flag = - self.delta_distance
            if path_flag > 0.0:
                path_reward = 0.1
            elif path_flag < 0.0:
                path_reward = -1.0
            else:
                path_reward = -0.01
            
            # Calculate miss penalty
            last_dist = dist - self.delta_distance
            if (last_dist < 1.0) and (- self.delta_distance > 0.0):
                miss_penalty = (last_dist - 1.0) * 2.0
            # Calculate human interaction reward if not in test local environment
        if consider_human:
            # Check if the agent has crashed
            crashed = self.ob['isCrashed']
            if crashed:
                human_reward = -2.0  # Negative reward for crashing
            else:
                human_reward = 0.0
        final_reward = self.get_final_reward(target_reward, path_reward, miss_penalty, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, miss_penalty, human_reward, strategy_name='reward_strategy_6')


    def reward_strategy_7(self, consider_human, reward_type):
        # NOTE: All compare to reward 6 
        # remove step reward from strategy 5
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        step_reward = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method

        # Get the current distance from the observation
        dist = self.ob['distance']
        epsilon = 0.1

        if consider_human:
            # Check if the agent has crashed
            crashed = self.ob['isCrashed']
            if crashed:
                human_reward = -1.0  # Negative reward for crashing
            else:
                human_reward = 6/(dist+epsilon)
        
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 3.0
            else:
                target_reward = -3.0
        else:
            # Calculate path fidelity reward
            path_flag = - self.delta_distance
            if path_flag > 0.0:
                path_reward = 1
            elif path_flag < 0.0:
                path_reward = -1
            else:
                path_reward = -0.01
            

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_7')


    def reward_strategy_8(self, consider_human, reward_type):
        # remove step reward from strategy 5
        # Initialize rewards and penalties
        target_reward = 0.0
        path_reward = 0.0
        step_reward = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method

        # Get the current distance from the observation
        dist = self.ob['distance']
        epsilon = 0.1

        # Check if the agent has stopped
        if consider_human:
            # Check if the agent has crashed
            crashed = self.ob['isCrashed']
            if crashed:
                if self.isCrashed_record[-1]:
                    human_reward = -2.0  # Negative reward for crashing
                else:
                    human_reward = -4.0  # Negative reward for crashing
            else:
                if self.isCrashed_record[-1]:
                    human_reward = 12/(dist + epsilon)  # Negative reward for crashing
                else:
                    human_reward = 6/(dist + epsilon)
                if self.delta_distance == 0:
                    human_reward = 0
            self.isCrashed_record.append(crashed)
        
        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 3.0
            else:
                target_reward = -3.0
        else:
            # Calculate path fidelity reward
            path_flag = - self.delta_distance
            if path_flag > 0.0:
                path_reward = 1
            elif path_flag < 0.0:
                path_reward = -1.0
            else:
                path_reward = -0.5

        final_reward = self.get_final_reward(target_reward, path_reward, step_reward, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, step_reward, human_reward, strategy_name='reward_strategy_8')
