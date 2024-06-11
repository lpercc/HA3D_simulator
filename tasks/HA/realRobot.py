import os
HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")
import argparse
import sys
import utils
import numpy as np
import random
import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math
from model import EncoderLSTM, AttnDecoderLSTM
sys.path.append(HA3D_SIMULATOR_PATH)
from scripts.video_feature_loader import TimmExtractor
from utils import read_vocab,Tokenizer,padding_idx

MODEL_NAME = "resnet152.a1_in1k"
FPS = 16
FEATURE_SIZE = 2048
TRAIN_VOCAB = os.path.join(HA3D_SIMULATOR_PATH, 'tasks/HA/data/train_vocab.txt')
TRAINVAL_VOCAB = os.path.join(HA3D_SIMULATOR_PATH, 'tasks/HA/data/trainval_vocab.txt')
MAX_INPUT_LENGTH = 80

class AgentLLA():
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
    feedback_options = ['argmax', 'sample']

    def __init__(self, robot, results_path, encoder, decoder, episode_len=20):
        self.env = robot
        self.results_path = results_path
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len

    @staticmethod
    def n_inputs():
        return len(AgentLLA.model_actions)

    @staticmethod
    def n_outputs():
        return len(AgentLLA.model_actions)-2 # Model doesn't output start or ignore


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

    def inference(self, scanId, viewpointId):
        obs = np.array(self.env.reset(scanId, viewpointId))
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        env_action = [None] * batch_size
        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs) # Image features from obs
            h_t,c_t,alpha,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')

            probs = F.softmax(logit, dim=1)
            m = D.Categorical(probs)
            a_t = m.sample()            # sampling an action from model

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
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # Early exit if all ended
            if ended.all():
                break
        return traj

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.encoder.eval()
        self.decoder.eval()

class UnitreeRobot():
    ''' Implements the HA sim2real task, using Unitree four legs robot'''
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
    step_ = 0
    scanId = ''
    viewpointId = ''
    def __init__(self, instr_id, instr, tokenizer, extractor):
        
        self.extractor = extractor
        #TimmExtractor(model_name=MODEL_NAME, fps=FPS, device=device)
        self.instruction = {}
        self.instruction['instr_id'] = instr_id
        self.instruction['instruction'] = instr
        self.instruction['instr_encoding'] = tokenizer.encode_sentence(instr)

    def newEpisode(self, scanId, viewpointId):
        self.step_ = 0
        self.scanId = scanId
        self.viewpointId = viewpointId

    def get_front_camera(self,FPS):
        frames = np.random.randint(0, 256, (FPS, 480, 640, 3), dtype='uint8')
        return frames

    def get_IMU():
        heading = 0
        elevation = 0
        print(f"heading,elevation")
        return heading, elevation

    def getState(self):
        state = RobotState()
        state.step = self.step_
        state.scanId = self.scanId
        state.location.viewpointId = self.viewpointId

        ''' Get list of states augmented with precomputed image features. rgb field will be empty. '''
        frames = self.get_front_camera(FPS)
        assert frames.shape[0] == FPS
        self.extractor.load_video(frames)
        feature = self.extractor.extract_features().squeeze()
        #print(feature.shape)
        return feature, state

    def makeAction(self, action):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        index = self.env_actions.index(action[0])
        print(f"Robot action: {self.model_actions[index]}")

    def _get_obs(self):
        feature,state = self.getState()
        ob = {
            'instr_id' : self.instruction['instr_id'],
            'scan' : state.scanId ,
            'viewpoint' : state.location.viewpointId,
            'heading' : state.heading,
            'elevation' : state.elevation,
            'feature' : feature,
            'step' : state.step,
            'navigableLocations' : state.navigableLocations,
            'instr_encoding' : self.instruction['instr_encoding'],
        }
        return [ob]

    def reset(self,scanId,viewpointId):
        ''' Load a new minibatch / episodes. '''
        self.newEpisode(scanId, viewpointId)
        return self._get_obs()

    def step(self, action):
        ''' Take action (same interface as makeActions) '''
        self.step_ += 1
        self.makeAction(action)
        return self._get_obs()

class RobotState():
    def __init__(self):
        locations = []
        location = {"viewpointId" : "0",
            "rel_heading" : 0,
            "rel_elevation" : 0,
            "rel_distance" : 0
            }
        self.scanId = "9424"
        self.step = 0
        self.location = Location(location)
        self.heading = 0
        self.elevation = 0
        for i in range(8):
            location = {"viewpointId" : f"{i+1}",
            "rel_heading" : math.radians(i*45),
            "rel_elevation" : 0,
            "rel_distance" : 0.5
            }
            locations.append(Location(location))
        self.navigableLocations = locations

class Location():
    def __init__(self,location):
        self.viewpointId = location["viewpointId"]
        self.rel_heading = location["rel_heading"]
        self.rel_elevation = location["rel_elevation"]
        self.rel_distance = location["rel_distance"]


def main(args):
    
    instr_id = '001'
    instr = "Go ahead and stop at the water fountain."
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    max_episode_len = 20
    word_embedding_size = 256
    action_embedding_size = 32
    hidden_size = 512
    bidirectional = False
    dropout_ratio = 0.5
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    results_path = ''
    encoder_path = os.path.join(HA3D_SIMULATOR_PATH, 'tasks/HA/snapshots/seq2seq_sample_imagenet_train_enc_iter_20000')
    decoder_path = os.path.join(HA3D_SIMULATOR_PATH, 'tasks/HA/snapshots/seq2seq_sample_imagenet_train_dec_iter_20000')
    train_vocab = read_vocab(TRAIN_VOCAB)
    tokenizer = Tokenizer(vocab=train_vocab, encoding_length=MAX_INPUT_LENGTH)
    extractor = TimmExtractor(model_name=MODEL_NAME, fps=FPS, device=device)
    realRobot = UnitreeRobot(instr_id, instr, tokenizer, extractor)
    
    encoder = EncoderLSTM(len(train_vocab), word_embedding_size, enc_hidden_size, padding_idx,
                dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(AgentLLA.n_inputs(), AgentLLA.n_outputs(),
                action_embedding_size, hidden_size, dropout_ratio).cuda()
    
    agent = AgentLLA(realRobot, results_path, encoder, decoder, max_episode_len)
    agent.load(encoder_path, decoder_path)
    print('start inference')
    agent.inference(scanId='9424', viewpointId='942401')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='ID for the cuda')
    args = parser.parse_args
    main(args)