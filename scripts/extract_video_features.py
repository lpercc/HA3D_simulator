#!/usr/bin/env python3
# The document is modified from the original file in the following link: https://github.com/peteanderson80/Matterport3DSimulator/blob/master/scripts/precompute_img_features.py

""" Script to precompute image features using a Caffe ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. """

import sys
sys.path.append('./')
from tqdm import tqdm
import HA3DSim
import numpy as np
import json
import math
import base64
import csv
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

# import out video_feature_loader.py
from video_feature_loader import TimmExtractor

csv.field_size_limit(sys.maxsize)


# Caffe and MatterSim need to be on the Python path

TSV_FIELDNAMES = ["scanId", "viewpointId", "image_w", "image_h", "vfov", "features"]
VIEWPOINT_SIZE = 36  # Number of discretized views from one viewpoint
FEATURE_SIZE = 2048 # FEATURE SIZE will change corresponding to certain model
#BATCH_SIZE = 4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
MODEL_NAME = "resnet152.a1_in1k"
#WEIDHTS_KEY = "IMAGENET1K_V1"
#VIDEO_LEN = 60
#FPS = 16
FPS = 1
#OUTFILE = "img_features/ResNet-152-imagenet_60.tsv"
OUTFILE = f"img_features/ResNet-152-imagenet_{FPS}_ALL.tsv"
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

'''
# 原本用于 Caffe 的图像预处理, 但是现在我们使用 PyTorch, 保留以参考
def transform_img(im):
    """Prep opencv 3 channel image for the network"""
    im = np.array(im, copy=True)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= np.array([[[103.1, 115.9, 123.2]]])  # BGR pixel mean
    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, :, :, :] = im_orig
    blob = blob.transpose((0, 3, 1, 2))
    return blob
'''


def build_tsv(args):
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
    device = 'cuda:'+args.gpu if torch.cuda.is_available() else 'cpu'
    
    # init a extractor
    # here we use a resnet 152 B as feature extractor, the output feature will be (2048,) 
    extractor = TimmExtractor(model_name=MODEL_NAME, fps=FPS, device=device)

    tsv_path = f"{OUTFILE.split('.')[0]}_{viewpoint_s}-{viewpoint_e}.tsv"
    
    with open(tsv_path, "a") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        # Loop all the viewpoints in the simulator
        print(f"viewpoint:{viewpoint_s}--{viewpoint_e}")
        data = read_tsv(tsv_path)
        all_viewpointIds = load_viewpointids()
        if len(data) > 0:
            print(data[-1]["scanId"], data[-1]["viewpointId"])
            print(all_viewpointIds[viewpoint_s+len(data)-1])
            assert (data[-1]["scanId"],data[-1]["viewpointId"]) == all_viewpointIds[viewpoint_s+len(data)-1]
        viewpointIds = all_viewpointIds[viewpoint_s+len(data):viewpoint_e]
        for _, (scanId, viewpointId) in enumerate(viewpointIds, start=len(data)):
            # Loop all discretized views from this location
            features = np.empty([VIEWPOINT_SIZE, FEATURE_SIZE], dtype=np.float32)
            bar = tqdm(range(VIEWPOINT_SIZE))
            for ix in bar:
                #print(f'ix {ix}')
                bar.set_description()
                if ix == 0:
                    sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    sim.makeAction([0], [1.0], [1.0])
                else:
                    sim.makeAction([0], [1.0], [0])

                state = sim.getState(FPS)[0]
                assert state.viewIndex == ix

                # Transform and save generated image
                video = state.video  # Pseudo code, need to correct this #TODO
                
                assert video.shape == (FPS, HEIGHT, WIDTH, 3)

                # extractor should load_video first to get video 
                # video should be a numpy array with shape (F, W, H, C)
                extractor.load_video(video)
                feature = extractor.extract_features()
                #print(f"feature shape: {feature.shape}")
                # the output features should be a numpy adarry with size (FEATURE_SIZE, )
                features[ix, :] = feature
                
                bar.set_description(f"Processing {_}th view point with {ix} decrete view.")

            writer.writerow(
                {
                    "scanId": scanId,
                    "viewpointId": viewpointId,
                    "image_w": WIDTH,
                    "image_h": HEIGHT,
                    "vfov": VFOV,
                    "features": str(base64.b64encode(features.tobytes()), "utf-8"), #TODO 
                }
            )


def read_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for item in reader:
            item["scanId"] = item["scanId"]
            item["viewpointId"] = item["viewpointId"]
            item["image_h"] = int(item["image_h"])
            item["image_w"] = int(item["image_w"])
            item["vfov"] = int(item["vfov"])
            item["features"] = np.frombuffer(
                base64.b64decode(item["features"]), dtype=np.float32
            ).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
            in_data.append(item)
    return in_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--viewpoint_s', default=0)
    parser.add_argument('--viewpoint_e', default=-1)
    args = parser.parse_args()
    build_tsv(args)
    tsv_path = f"{OUTFILE.split('.')[0]}_{int(args.viewpoint_s)}-{int(args.viewpoint_e)}.tsv"
    data = read_tsv(tsv_path)
    print("Completed %d viewpoints" % len(data))