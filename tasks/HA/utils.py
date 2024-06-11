''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
import random

HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")
# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(HA3D_SIMULATOR_PATH, 'connectivity/%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def load_datasets(splits):
    data = []
    for split in splits:
        assert split in ['train', 'val_seen', 'val_unseen', 'test']
        with open(os.path.join(HA3D_SIMULATOR_PATH, 'tasks/HA/data/HA_%s.json' % split)) as f:
            data += json.load(f)
    random.seed(10)
    random.shuffle(data)
    return data


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        for word in self.split_sentence(sentence)[::-1]: # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))
        return np.array(encoding[:self.encoding_length])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::-1]) # unreverse before output


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab

# TODO: 整体最优策略
def remove_global_nodes_and_find_path(G, humanLocations, currentViewpointId, goalViewpointId):
    # deleteAll = False 把最近的人从图中删除，deleteAll = True 把所有人从图中删除
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
    # 创建要删除的节点列表
    nodes_to_remove = []
    # 创建图的副本以避免修改原始图
    G_copy = G.copy()
    # 遍历每个人的位置
    for humanLocation in humanLocations:
        # 遍历图中的每个节点
        for node, data in G_copy.nodes(data=True):
            if node == currentViewpointId or node == goalViewpointId or node in nodes_to_remove:
                # 如果节点是当前位置或目标位置，跳过不删除
                continue
            node_position = data['position']
            # 计算节点与人的欧几里得距离
            if euclidean_distance(humanLocation, node_position[:3]) < 1:
                # 如果距离小于1，标记该节点以便稍后从图中移除
                nodes_to_remove.append(node)
    # 在迭代完成后移除标记的节点
    for node in nodes_to_remove:
        G_copy.remove_node(node)

    # 尝试找到从当前位置到目标位置的最短路径
    try:
        path = nx.shortest_path(G_copy, source=currentViewpointId, target=goalViewpointId)
    except nx.NetworkXNoPath:
        # 如果没有找到路径，返回当前位置
        path = [currentViewpointId, currentViewpointId]
    return path

# TODO: 局部最优策略
def remove_local_nodes_and_find_path(G, humanLocations, currentViewpointId, goalViewpointId, radius):
    # deleteAll = False 把最近的人从图中删除，deleteAll = True 把所有人从图中删除
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
    # 创建要删除的节点列表
    nodes_to_remove = []
    # 创建图的副本以避免修改原始图
    G_copy = G.copy()
    
    humanInfo = []
    agentLocation = G_copy.nodes[currentViewpointId]['position']
    for humanLocation in humanLocations:
        humanDistance  = euclidean_distance(humanLocation, agentLocation[:3])
        # 只考虑agent与人之间的距离小于4.5的
        if humanDistance < radius:
            humanInfo.append((humanDistance, humanLocation))
    #humanInfo.sort(key=lambda x:x[0]) 
    
    
    # 遍历每个人的位置
    for (humanDistance, humanLocation) in humanInfo:
        # 遍历图中的每个节点
        for node, data in G_copy.nodes(data=True):
            if node == currentViewpointId or node == goalViewpointId or node in nodes_to_remove:
                # 如果节点是当前位置或目标位置，跳过不删除
                continue
            node_position = data['position']
            # 计算节点与人的欧几里得距离
            if euclidean_distance(humanLocation, node_position[:3]) < 1:
                # 如果距离小于1，标记该节点以便稍后从图中移除
                nodes_to_remove.append(node)
    # 在迭代完成后移除标记的节点
    for node in nodes_to_remove:
        G_copy.remove_node(node)

    # 尝试找到从当前位置到目标位置的最短路径
    try:
        path = nx.shortest_path(G_copy, source=currentViewpointId, target=goalViewpointId)
    except nx.NetworkXNoPath:
        # 如果没有找到路径，返回当前位置
        path = [currentViewpointId, currentViewpointId]
    return path

def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    vocab = []
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def relHumanAngle(humanLocations, agentLocation, agentHeading, agentElevation):
    nearestHuman = []
    minDistance = 1000
    for humanLocation in humanLocations:
        distance = np.linalg.norm(np.array(humanLocation) - np.array(agentLocation))
        if distance < minDistance:
            minDistance = distance
            nearestHuman = humanLocation
    heading_angle, elevation_angle = horizontal_and_elevation_angles(agentLocation, nearestHuman)
    return heading_angle-agentHeading, elevation_angle-agentElevation, minDistance

def horizontal_and_elevation_angles(point1, point2):
    """
    计算两个3D坐标之间的相对水平夹角和仰角（俯仰角）
    :param point1: 第一个3D坐标
    :param point2: 第二个3D坐标
    :return: 相对水平夹角和仰角的弧度表示
    """
    vector = np.array(point2) - np.array(point1)
    horizontal_angle = np.arctan2(vector[0], vector[1])
    elevation_angle = np.arctan2(vector[2], np.linalg.norm(vector[:2]))
    return horizontal_angle, elevation_angle
    
def check_agent_status(traj, max_steps=30, ended=True):
    # TODO: use this function to check all trajs agent status
    # DONE: ADD functions to check rewards
    print(f'{"=" * 10}Checking agent status...{"=" * 10}')
    table = PrettyTable()
    table.field_names = ['Description', 'Status']
    table.add_row(['Number of episodes', len(traj)])
    table.add_row(['Episode Length', max_steps])
    
    # Whether terminate navigaton early or not 
    terminate = False
    terminate_count = 0 
    
    # Whether agent still has steps to go
    still_go = False
    still_go_count = 0
    for i, ep in enumerate(traj):
        if len(traj[i]['unique_path']) < 6:
            terminate = True
            terminate_count += 1
            
        if not ended[i]:
            still_go = True
            still_go_count += 1
            
    
    table.add_row(['Terminate Navigation Early', terminate])
    table.add_row(['Number of Terminations', terminate_count])
    
    table.add_row(['Agent Still Has Steps to Go', still_go])
    table.add_row(['Number of Agents Still Going', still_go_count])
    
    final_rewards = []
    target_rewards = []
    path_rewards = []
    miss_penalties = []
    human_rewards = []
    
    for i in traj: 
        final_rewards.append(i['final_reward'])
        target_rewards.append(i['target_reward'])
        path_rewards.append(i['path_reward'])
        miss_penalties.append(i['missing_reward'])
        human_rewards.append(i['human_reward'])    
    
    """final_rewards_array = np.array(final_rewards)
    target_rewards_array = np.array(target_rewards)
    path_rewards_array = np.array(path_rewards)
    miss_penalties_array = np.array(miss_penalties)
    human_rewards_array = np.array(human_rewards)
    
    plot_rewards(final_rewards, 'Final Rewards')
    plot_rewards(target_rewards, 'Target Rewards')
    plot_rewards(path_rewards, 'Path Rewards')
    plot_rewards(miss_penalties, 'Miss Penalties')
    plot_rewards(human_rewards, 'Human Rewards')"""
    
    # plot reward in lineplot 
    print(table)
        
def plot_rewards(rewards, reward_name): 
    # DONE: plot rewards in lineplot 
    print(f'{"=" * 10}Plotting rewards...{"=" * 10}')
    
    plt.plot(rewards)
    plt.xlabel('steps')
    plt.ylabel('Rewards')
    plt.title(f'{reward_name} in Each Episode')
    plt.show()
    plt.close()


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
        # Experiment Group 1
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
        if reward_type == 'sparse':
            final_reward = target_reward + human_reward
        elif reward_type == 'dense':
            final_reward = target_reward + path_reward + miss_penalty + human_reward
        return final_reward
    
    def reward_strategy_1(self, reward_type):
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
        # TODO: We only use Target for strategy one
        target_reward = 0.0
        path_reward = 0.0
        miss_penalty = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method

        # Get the current distance from the observation
        dist = self.ob['distance']

        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 5.0
            else:
                target_reward = -5.0

        final_reward = self.get_final_reward(target_reward, path_reward, miss_penalty, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, miss_penalty, human_reward, strategy_name='reward_strategy_1')    
        
        # Get the current distance from the observation
        dist = self.ob['distance']

        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 0
            else:
                target_reward = 0
        
        # Human Reward 
        crashed = self.ob['isCrashed']
        if crashed: 
            human_reward = -2.0 

        final_reward = self.get_final_reward(target_reward, path_reward, miss_penalty, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, miss_penalty, human_reward, strategy_name='reward_strategy_3')
        
        
        
    def reward_strategy_2(self, reward_type):
    # Initialize rewards and penalties
        # TODO: We only use Target for strategy one
        target_reward = 0.0
        path_reward = 0.0
        miss_penalty = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method

        # Get the current distance from the observation
        dist = self.ob['distance']

        # Check if the agent has stopped
        if self.action == (0, 0, 0):
            if dist < 3.0:
                target_reward = 5.0
            else:
                target_reward = -5.0
        
        # Human Reward 
        crashed = self.ob['isCrashed']
        if crashed: 
            human_reward = -2.0 

        final_reward = self.get_final_reward(target_reward, path_reward, miss_penalty, human_reward, reward_type)
        self.append_rewards(final_reward, target_reward, path_reward, miss_penalty, human_reward, strategy_name='reward_strategy_2')
        
    def reward_strategy_3(self, reward_type):
    # Initialize rewards and penalties
        # TODO: We only use Target for strategy one
        target_reward = 0.0
        path_reward = 0.0
        miss_penalty = 0.0
        human_reward = 0.0  # Human interaction reward is calculated in the calculate method
