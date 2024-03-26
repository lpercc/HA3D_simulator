''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
sys.path.append('./')
import HC3DSim
import csv
import numpy as np
import math
import base64
import os
import random
import networkx as nx
from scripts.video_feature_loader import TimmExtractor 
import torch
from utils import load_datasets, load_nav_graphs, relHumanAngle

from tqdm import tqdm 

MODEL_NAME = "resnet152.a1_in1k"
FPS = 16
csv.field_size_limit(sys.maxsize)
MODELING_ONLY = True


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        if feature_store is not None :
            print('Loading image features from %s' % feature_store)
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
            self.features = {}
            with open(feature_store, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                for item in reader:
                    self.image_h = int(item['image_h'])
                    self.image_w = int(item['image_w'])
                    self.vfov = int(item['vfov'])
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    self.features[long_id] = np.frombuffer(base64.b64decode(item['features']),
                            dtype=np.float32).reshape((36, 2048))
            self.renderingFlag = False
        else:
            print('Image features Extractor Timm')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
            self.renderingFlag = True
            batch_size = 1
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.extractor = TimmExtractor(model_name=MODEL_NAME, fps=FPS, device=device)
        
        self.batch_size = batch_size
        if not MODELING_ONLY:
            dataset_path = os.path.join(os.environ.get("HC3D_SIMULATOR_DTAT_PATH"), "data/v1/scans")
            self.sim.setDatasetPath(dataset_path)
        self.sim = HC3DSim.HCSimulator()
        self.sim.setRenderingEnabled(self.renderingFlag)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setBatchSize(self.batch_size)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.sim.setDepthEnabled(True)
        self.sim.initialize()

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        self.sim.newEpisode(scanIds, viewpointIds, headings, [0]*self.batch_size)

    def getStates(self):
        ''' Get list of states augmented with precomputed image features. rgb field will be empty. '''
        feature_states = []
        if self.renderingFlag:
            state, frames = self.sim.getStepState(FPS)
            assert frames.shape == (FPS, self.image_h, self.image_w, 3)
            self.extractor.load_video(frames)
            feature = self.extractor.extract_features().squeeze()
            #print(feature.shape)
            feature_states.append((feature, state))
        else:
            for state in self.sim.getState():
                long_id = self._make_id(state.scanId, state.location.viewpointId)
                if self.features:
                    feature = self.features[long_id][state.viewIndex,:]
                    feature_states.append((feature, state))
                else:
                    feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        ix = []
        heading = []
        elevation = []
        for i,h,e in actions:
            ix.append(int(i))
            heading.append(float(h))
            elevation.append(float(e))
        self.sim.makeAction(ix, heading, elevation)

    def makeSimpleActions(self, simple_indices):
        ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. '''
        actions = []
        for i, index in enumerate(simple_indices):
            if index == 0:
                actions.append((1, 0, 0))
            elif index == 1:
                actions.append((0,-1, 0))
            elif index == 2:
                actions.append((0, 1, 0))
            elif index == 3:
                actions.append((0, 0, 1))
            elif index == 4:
                actions.append((0, 0,-1))
            else:
                sys.exit("Invalid simple action");
        self.makeActions(actions)


class HCBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None, text_embedding_model=None, device='cpu'):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.data = []
        self.scans = []
        
        # tqdm bar 
        bar = tqdm(load_datasets(splits))
        for item in bar: #TODO: change load datasets to load from pickle word embedding file simultaneously
            # Split multiple instructions into separate entries
            bar.set_description(f"Loading {item['scan']}, Use text_embedding_model? {text_embedding_model}") # use for check if we load same scam multiple times
            for j,instr in enumerate(item['instructions']):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                if tokenizer and not text_embedding_model: # 
                    new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                elif tokenizer and text_embedding_model:
                    with torch.no_grad(): 
                        inputs = tokenizer(instr, return_tensors="pt")
                        inputs = inputs.to(device)
                        text_embedding_model = text_embedding_model.to(device)
                        outputs = text_embedding_model(**inputs)
                        new_item['instr_encoding'] = inputs['input_ids'].squeeze(0).cpu().numpy()
                        new_item['instr_embedding'] = outputs[0][:, 0, :].squeeze(0).cpu().numpy()
                self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        # TODO: remember to shuffle data
        #random.seed(self.seed)
        #random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        print('HCBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0
        
        
    def _get_human_distance(self, state): 
        ''' Get the distance between human and goal viewpoint. '''
        humanLocations = state.humanState
        # compute the nearest human relative heading and elevation
        relHeading, relElevation, minDistance = relHumanAngle(humanLocations, 
                                                              [state.location.x, state.location.y, state.location.z], 
                                                              state.heading,
                                                              state.elevation)
        return minDistance

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training.

        Args:
            state (State): The current state of the environment.
            goalViewpointId (str): The ID of the goal viewpoint.

        Returns:
            tuple: A tuple representing the next action to take. The tuple contains three values:
                - The change in x-coordinate (int)
                - The change in y-coordinate (int)
                - The change in z-coordinate (int)
        '''
        # human Location of one building
        # state.humanState:[[x1, y1, z1], [x2, y2, z2], ...]
        humanLocations = state.humanState
        # compute the nearest human relative heading and elevation
        relHeading, relElevation, minDistance = relHumanAngle(humanLocations, 
                                                              [state.location.x, state.location.y, state.location.z], 
                                                              state.heading,
                                                              state.elevation)

        if state.location.viewpointId == goalViewpointId:
            return (0, 0, 0) # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        

        # find next Viewpoint to avoid human
        if minDistance < 1.5 and abs(relHeading)<math.pi/6.0 and abs(relElevation)<math.pi/6.0:
            for loc in state.navigableLocations:
                _, _, Distance = relHumanAngle(humanLocations, 
                                                    [loc.x, loc.y, loc.z], 
                                                    loc.rel_heading,
                                                    loc.rel_elevation)
                if Distance > 1.5 and abs(loc.rel_heading-relHeading) < 2*math.pi/3.0 and abs(loc.rel_heading-relHeading) > math.pi/3.0:
                    nextViewpointId = loc.viewpointId
                    break
                else:
                    nextViewpointId = path[1]
        else:
            nextViewpointId = path[1]    
        # Can we see the next viewpoint?
        for i,loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0) # Turn right
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0) # Turn left
                elif loc.rel_elevation > math.pi/6.0 and state.viewIndex//12 < 2:
                      return (0, 0, 1) # Look up
                elif loc.rel_elevation < -math.pi/6.0 and state.viewIndex//12 > 0:
                      return (0, 0,-1) # Look down
                else:
                      return (i, 0, 0) # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex//12 == 0:
            return (0, 0, 1) # Look up
        elif state.viewIndex//12 == 2:
            return (0, 0,-1) # Look down
        # Otherwise decide which way to turn
        pos = [state.location.x, state.location.y, state.location.z]
        target_rel = self.graphs[state.scanId].nodes[nextViewpointId]['position'] - pos
        target_heading = math.pi/2.0 - math.atan2(target_rel[1], target_rel[0]) # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0*math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0,-1, 0) # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0,-1, 0) # Turn left
        return (0, 1, 0) # Turn right

    def _get_obs(self):
        obs = []
        for i,(feature,state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'step' : state.step,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            if 'instr_embedding' in item:
                obs[-1]['instr_embedding'] = item['instr_embedding']
                obs[-1]['state_features'] = np.concatenate([feature, item['instr_embedding']]) # 2048 + 768 = 2816 feature size
            # add distance bewteen agent and goal viewpoint 
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
            obs[-1]['human_distance'] = self._get_human_distance(state)
        return obs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch()
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()
