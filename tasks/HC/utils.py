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
HC3D_SIMULATOR_PATH = os.environ.get("HC3D_SIMULATOR_PATH")
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
        with open(os.path.join(HC3D_SIMULATOR_PATH, 'connectivity/%s_connectivity.json' % scan)) as f:
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
        with open(os.path.join(HC3D_SIMULATOR_PATH, 'tasks/HC/data/HC_%s.json' % split)) as f:
            data += json.load(f)
    random.seed(10)
    random.shuffle(data)
    # 按照每个字典中 'scan' 键的值对数据进行排序
    sorted_data = sorted(data, key=lambda x: x['scan'])
    
    return sorted_data


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

def check_agent_status(traj, max_steps, ended):
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
    
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('steps')
    plt.ylabel('Rewards')
    plt.title(f'{reward_name} in Each Episode')
    plt.show()
    plt.close()

def calculate_rewards(ob, action, delta_distance, reward_type='dense', test_local=True): 
    # Calculate rewards besed on recent ob
    # 需要当前的 Scan ID, 以及目前所在的最短 Path
    # Scan 用于判断目前人的状态
    # We need to know distance 
    # 目标的 Path ID 用于对比 Grounding Truth 的 Rewards 
    # 还需要下一步的观察值
    # DONE: 设计两种不同的 Reward 模式, 一个应该是稀疏的 (只与终点有关), 另一个是稠密的 (与当前位置有关)
    recent_action = action
    dist = ob['distance']
    
    target_reward = 0.0 
    path_reward = 0.0
    miss_penalty = 0.0
    # if stop, then we give a target reward 
    if recent_action == (0, 0, 0):
        if dist < 3.0:
            target_reward = 3.0
        else:
            target_reward = - 3.0
    else: 
        # Path Fidelity Reward 
        path_flag =  - delta_distance
        if path_flag > 0.0: 
            path_reward = 1.0
        elif path_flag < 0.0: 
            path_reward = -1.0 
        else: 
            path_reward = - 0.1 # TODO: 这里可以考虑加入一个小的负值, 以防止 Agent 一直停留在原地
        
        # Miss the target penalty 
        last_dist = dist - delta_distance
        if (last_dist < 1.0) and (- delta_distance > 0.0): 
            miss_penalty =(1.0 - last_dist) * 2.0
            
    human_reward = 0.0
    if not test_local:
        # Now we calculate human related reward 
        # TODO: there are two choices here, one is just considering the distance between human and agent, the second is that considering the avoid(action) step. 
        # Now we calculate first one. 
        human_distance = ob['human_distance']
        if human_distance < 2.25: 
            human_reward = - 2.0 
        elif 2.25 < human_distance < 4.5: 
            human_reward = 2.0 
        else: 
            human_reward = 0.0
            
        # for second, we calculate use the nearest viewpoint to the human location. 
        # Now we calculate the avoid step reward
        # # TODO: if there is a human in next step, and the action is avoid like action , then we give a reward. (We just use teacher action or use rewards that copy teacher actions in Reccurent BERT))
        # # The avaliable_navigation is a list of MatterSim.Viewpoint objects. The attribute of each object is viewpointId, x, y, z 
        
        
    if reward_type == 'sparse': 
        final_reward = target_reward + miss_penalty
    elif reward_type == 'dense':
        final_reward = target_reward + path_reward + miss_penalty + human_reward
        
    # DONE: output rewards separately
    
    return final_reward, target_reward, path_reward, miss_penalty, human_reward