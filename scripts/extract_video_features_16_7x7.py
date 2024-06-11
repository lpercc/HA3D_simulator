#!/usr/bin/env python3
# The document is modified from the original file in the following link: https://github.com/peteanderson80/Matterport3DSimulator/blob/master/scripts/precompute_img_features.py

""" Script to precompute image features using a Caffe ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. """

import sys
from tqdm import tqdm
sys.path.append('./')
import HA3DSim
import numpy as np
import json
import math
import argparse
import torchvision 
import torchvision.models as models
import torchvision.transforms as transforms
import tempfile
import os 
import torch
from pathlib import Path
#from tqdm import tqdm 
import torch.nn as nn
from urllib.request import urlretrieve
import pickle
# import out video_feature_loader.py
from video_feature_loader import TimmExtractor


# Caffe and MatterSim need to be on the Python path

VIEWPOINT_SIZE = 36  # Number of discretized views from one viewpoint
FEATURE_SIZE = 2048 # FEATURE SIZE will change corresponding to certain model
#BATCH_SIZE = 4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
MODEL_NAME = "resnet152.a1_in1k"
#WEIDHTS_KEY = "IMAGENET1K_V1"
GAP = 1
FPS = 16
VIDEO_LEN = 80
#FPS = 1
#OUTFILE = "img_features/ResNet-152-imagenet_60.tsv"
OUTFILE = f"ResNet-152-imagenet_{VIDEO_LEN}_{FPS}_2048x7x7.tsv"
#OUTFILE = "img_features/ResNet-152-imagenet_1.tsv"
GRAPHS = "connectivity/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Simulator image parameters
WIDTH = 640
HEIGHT = 480
VFOV = 60


def load_viewpointids():
    viewpointIds = []
    with open(GRAPHS + "scans.txt") as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS + scan + "_connectivity.json") as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpointIds.append((scan, item["image_id"]))
    print("Loaded %d viewpoints" % len(viewpointIds))
    return viewpointIds

def build_pkl(args):
    # Set up the simulator
    viewpoint_s = int(args.viewpoint_s)
    viewpoint_e = int(args.viewpoint_e)
    dataset_path = os.path.join(os.environ.get("HA3D_SIMULATOR_DATA_PATH"), "data/v1/scans")
    # Create child processes
    from multiprocessing import Process
    def runProgram(command, suppress_output=False):
        if suppress_output:
            command += " >/dev/null 2>&1"
        print(command)
        os.system(f'python {command}')
    Process(target=runProgram, args=(f"HA3DRender.py --pipeID {args.pipeID}", False)).start()

    sim = HA3DSim.HASimulator(pipeID = args.pipeID)
    sim.setRenderingEnabled(True)
    sim.setDatasetPath(dataset_path)
    sim.setDepthEnabled(True)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()
 
    # set up device, will we use this?
    device = 'cuda:'+args.cuda if torch.cuda.is_available() else 'cpu'
    # init a extractor
    # here we use a resnet 152 B as feature extractor, the output feature will be (2048,) 
    #extractor = TimmExtractor(model_name=MODEL_NAME, fps=VIDEO_LEN, device=device)
    extractor = TimmExtractor(model_name=MODEL_NAME, fps=int(VIDEO_LEN/FPS), device=device, fuse='mean')
    
    pkl_path1 = os.path.join(args.img_feat, f"{OUTFILE.split('.')[0]}_mean_{viewpoint_s}-{viewpoint_e}.pkl")
    # Loop all the viewpoints in the simulator
    # 打开Pickle文件
    if os.path.exists(pkl_path1):
        try:
            with open(pkl_path1, 'rb') as file:
                data1 = pickle.load(file)
        except EOFError:
            print("Pickle文件为空或损坏。")
            data1 = []  # 或者设置为默认值
    else:
        data1 = []
    print(f"viewpoint:{viewpoint_s}--{viewpoint_e}")
    all_viewpointIds = load_viewpointids()
    if len(data1) > 0:
        print(data1[-1]["scanId"], data1[-1]["viewpointId"])
        print(all_viewpointIds[viewpoint_s+len(data1)-1])
        assert (data1[-1]["scanId"],data1[-1]["viewpointId"]) == all_viewpointIds[viewpoint_s+len(data1)-1]
    viewpointIds = all_viewpointIds[viewpoint_s+len(data1):viewpoint_e]
    try:
        for _, (scanId, viewpointId) in enumerate(viewpointIds, start=len(data1)):
            # Loop all discretized views from this location
            features1 = np.empty([VIEWPOINT_SIZE, int(VIDEO_LEN/FPS), FEATURE_SIZE, 7, 7], dtype=np.float32)
            #assert _ < 1000
            bar = tqdm(range(VIEWPOINT_SIZE))
            for ix in bar:
                if ix == 0:
                    sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    sim.makeAction([0], [1.0], [1.0])
                else:
                    sim.makeAction([0], [1.0], [0])

                state, video= sim.getStepState(frames=VIDEO_LEN, gap=GAP)
                assert state.viewIndex == ix

                video_len = int(VIDEO_LEN/GAP)
                # Transform and save generated image
                assert video.shape == (video_len, HEIGHT, WIDTH, 3)
                
                # 初始化一个标志变量，假设所有帧起初都是相同的
                all_frames_same = True
                # 遍历视频的每一帧，检查相邻帧之间是否有差异
                for i in range(8, video.shape[0], 8):  # 从第二帧开始比较
                    # 如果当前帧和前一帧之间有任何差异，则设置标志为 False 并退出循环
                    if not np.array_equal(video[i], video[i-8]):
                        all_frames_same = False
                        break
                if all_frames_same:
                    extractor.load_video(video[0:int(VIDEO_LEN/FPS)])
                    feature = extractor.extract_features(keep_T=True)
                else:
                    extractor.load_video(video)
                    feature = extractor.extract_features(keep_T=True)
                assert feature.shape == (int(VIDEO_LEN/FPS), FEATURE_SIZE, 7, 7)
                features1[ix, :, :] = feature
                bar.set_description(f"Processing {_}th view point with {VIEWPOINT_SIZE} decrete view.")
            data1.append(
                {
                    "scanId": scanId,
                    "viewpointId": viewpointId,
                    "features": features1
                }
            )
    except Exception as e:
        print(f"error {e}")
    finally:
        with open(pkl_path1, 'wb') as file:
            pickle.dump(data1, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='0')
    parser.add_argument('--viewpoint_s', default=0)
    parser.add_argument('--viewpoint_e', default=10567)
    parser.add_argument('--img_feat', default='./')
    parser.add_argument('--pipeID', default=0)
    args = parser.parse_args()
    build_pkl(args)
    pkl_path1 = os.path.join(args.img_feat, f"{OUTFILE.split('.')[0]}_mean_{args.viewpoint_s}-{args.viewpoint_e}.pkl")
    with open(pkl_path1, 'rb') as file:
        data1 = pickle.load(file)
    print("Completed %d viewpoints" % len(data1))