''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import HABatch
from utils import load_datasets, load_nav_graphs
from agent import BaseAgent
sys.path.append('./')
HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")
HUMAN_VIEWPOINT = os.path.join(HA3D_SIMULATOR_PATH, 'human-viewpoint_pair/human_motion_text.json')
NEW_DATA = True


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans=None, tok=None):
        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for split in splits:
            for item in load_datasets([split]):
                if scans is not None and item['scan'] not in scans:
                    continue
                self.gt[str(item['path_id'])] = item
                self.scans.append(item['scan'])
                self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id
    
    def _get_human_distance(self, gt, path):
        ''' 计算当前路径上, 与人碰撞的次数, 则计为 1, 否则计为 0 
        path id 是当前 eval 的 path
        path 是之前的 trajectory. 
        '''
        hits = []
        flag = 0
        for i, ob in enumerate(path): 
            if ob[3]:
                hits.append(1)
                flag = 1
            else:
                hits.append(0)

        penalty = 0           
        for human in gt['human']:
            if human['human_rel_pos'] == 'Beginning' or human['human_rel_pos'] == 'End':
                penalty += 1
        return hits, flag, penalty
                
        # 我要每一次都做一次 loop 吗? 所有路径上看一遍有没有人? 
        # 这里需要知道当前路径的 Viewpoint ID, 然后索引这周边人的位置, 计算距离. 
        # 假设人 viewpointID 也在联通图中
        
    def _get_human_list(self, scan):
        with open(HUMAN_VIEWPOINT) as f: 
            human_point_list = json.load(f)
        return human_point_list[scan]
        
    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv, isCrashed] '''
        gt = self.gt[instr_id.split('_')[-2]] # 拥有这个 path id 的 item
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]    # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        
        
        if NEW_DATA:
            hits, hit, penalty = self._get_human_distance(gt, path)
            total_hits = sum(hits) - penalty
            self.scores['total_hits'].append(total_hits)
            self.scores['hit'].append(hit)
            
            
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_lengths'].append(
            self.distances[gt['scan']][start][goal]
        )

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                if 'trajectory' in item.keys():
                    self._score_item(item['instr_id'], item['trajectory'])
                elif 'path' in item.keys():
                    self._score_item(item['instr_id'], item['path'])
                else:
                    assert False, 'trajectory not in item'
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        if NEW_DATA:
            total_hits_rate = float(sum(self.scores['total_hits']))/float(len(self.scores['total_hits'])) # 新的 Metric, hits rate, 总共撞击的次数 / 总的运行次数
            hits_rate = float(sum(self.scores['hit']))/float(len(self.scores['hit'])) # 新的 Metric, hits rate, 总共撞击的次数 / 总的运行次数
    
            
            num_successes_nohits = len([i for i, h in zip(self.scores['nav_errors'], self.scores['total_hits']) if i < self.error_margin and h == 0]) # 新的 Metric, 没有撞击到人才算成功
            # 设置一个得分, 得分是 成功率 * 0.5 + 0.5 * (1 - hits_rate) * 成功率
            success_rate = num_successes / len(self.scores['nav_errors'])
            politeness_rate = success_rate * 0.8 + 0.2 * (1 - hits_rate) * success_rate
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            beta = np.std(np.array(self.scores['total_hits']))
            alpha = np.mean(np.array(self.scores['total_hits']))
            theta = 0.99
            scaler = -np.log(1/theta - 1) / beta
            weight = (1 - sigmoid(scaler * (np.array(self.scores['total_hits']) - alpha))) # weight 是一个 0-1 之间的值, 用来调整 hits 的影响
            weighted_errors = np.array(self.scores['nav_errors']) * weight + 0.5 * np.array(self.scores['nav_errors'])
            weighted_num_successes = len([i for i in weighted_errors if i < self.error_margin])
        
        
        
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        spls = []
        for err,length,sp in zip(self.scores['nav_errors'],self.scores['trajectory_lengths'],self.scores['shortest_path_lengths']):
            if err < self.error_margin:
                spls.append(sp/max(length,sp))
            else:
                spls.append(0)
        if NEW_DATA:
            score_summary ={
                'length': np.average(self.scores['trajectory_lengths']),
                'nav_error': np.average(self.scores['nav_errors']),
                'oracle success_rate': float(oracle_successes)/float(len(self.scores['oracle_errors'])),
                'success_rate': float(num_successes)/float(len(self.scores['nav_errors'])),
                'spl': np.average(spls),
                'total_hits_rate': total_hits_rate,
                'hits_rate': hits_rate, # 新的 Metric, hits rate, 单次撞击的次数的总和 / 总的运行次数
                'politeness_rate': politeness_rate,
                'successes_nohits_rate': float(num_successes_nohits)/float(len(self.scores['nav_errors'])),
                'hits_weighted_success_rate': float(weighted_num_successes)/float(len(self.scores['nav_errors'])),
                'trajectory steps': np.average(self.scores['trajectory_steps']),
            }
        else: 
            score_summary ={
                'length': np.average(self.scores['trajectory_lengths']),
                'nav_error': np.average(self.scores['nav_errors']),
                'oracle success_rate': float(oracle_successes)/float(len(self.scores['oracle_errors'])),
                'success_rate': float(num_successes)/float(len(self.scores['nav_errors'])),
                'spl': np.average(spls),
            }

        #assert score_summary['spl'] <= score_summary['success_rate']
        return score_summary, self.scores
            #assert len(self.scores['nav_errors']) == len(self.instr_ids)



    def bleu_score(self, path2inst):
        from bleu import compute_bleu
        refs = []
        candidates = []
        for path_id, inst in path2inst.items():
            path_id = str(path_id)
            assert path_id in self.gt
            # There are three references
            refs.append([self.tok.split_sentence(sent) for sent in self.gt[path_id]['instructions']])
            candidates.append([self.tok.index_to_word[word_id] for word_id in inst])

        tuple = compute_bleu(refs, candidates, smooth=False)
        bleu_score = tuple[0]
        precisions = tuple[1]

        return bleu_score, precisions


RESULT_DIR = os.path.join(HA3D_SIMULATOR_PATH, 'tasks/HA/results/')

def eval_simple_agents():
    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen', 'test']:
        env = HABatch(None, batch_size=1, splits=[split])
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_100.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_200.json'
    ]
    print('eval_seq2seq')
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':
    #eval_simple_agents()
    eval_seq2seq()
