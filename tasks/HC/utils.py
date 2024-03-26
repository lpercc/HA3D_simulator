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
        with open('connectivity/%s_connectivity.json' % scan) as f:
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
        with open('tasks/HC/data/HC_%s.json' % split) as f:
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
    """
    Check the status of the agent based on the trajectory, maximum steps, and ended flag.

    Args:
        traj (list): List of episodes containing the agent's trajectory.
        max_steps (int): Maximum number of steps allowed for each episode.
        ended (list): List of boolean values indicating whether each episode has ended.

    Returns:
        None

    Prints a table displaying various status information about the agent, including the number of episodes,
    episode length, whether navigation was terminated early, the number of terminations, whether the agent
    still has steps to go, and the number of agents still going.
    """
    print(f'{"=" * 10}Checking agent status...{"=" * 10}')
    table = PrettyTable() #TODO: Add PrettyTable exception
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
    
    print(table)
    
    
def calculate_rewards(ob, actions, reward_type='sparse',): 
    # Calculate rewards besed on recent ob
    # 需要当前的 Scan ID, 以及目前所在的最短 Path
    # Scan 用于判断目前人的状态
    # We need to know distance 
    # 目标的 Path ID 用于对比 Grounding Truth 的 Rewards 
    # 还需要下一步的观察值
    # TODO: 设计两种不同的 Reward 模式, 一个应该是稀疏的 (只与终点有关), 另一个是稠密的 (与当前位置有关)
    recent_action = actions[-1]
    
    if reward_type == 'sparse':
        if ob['next_location'] == ob['target_location']:
            return 1
        else:
            return 0
    
    
    raise NotImplementedError
        