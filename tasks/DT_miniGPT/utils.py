'''
Author: Dylan Li dylan.h.li@outlook.com
Date: 2024-03-26 15:23:17
LastEditors: Dylan Li dylan.h.li@outlook.com
LastEditTime: 2024-03-30 22:45:30
FilePath: /HA3D_simulator/tasks/HA/DT/utils.py
Description: 

Copyright (c) 2024 by Heng Li, All Rights Reserved. 
'''
"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import networkx as nx

HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")
# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def sample_from_logits(logits, temperature=1.0, sample=False, top_k=None):
    """
    Given a sequence of logits, predict the next token in the sequence,
    feeding the predictions back into the model each time. This function
    assumes that the logits are already produced by the model and are
    passed directly to it.
    """
    # Assuming logits are of shape (b, t, v) where b is batch size, t is sequence length, and v is vocabulary size
    # We only need the last logits for the next token prediction
    logits = logits[:, -1, :] / temperature
    
    # Optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    # Apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample from the distribution or take the most likely
    if sample:
        ix = torch.multinomial(probs, num_samples=1)
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)
    
    # Return the index of the sampled token
    return ix

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
